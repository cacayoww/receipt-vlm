"""
Receipt extractor with a model selector:
  baseline   = pretrained Qwen2.5-VL-3B, no fine-tuning
  finetuned  = baseline + LoRA adapter trained with TRL SFTTrainer (hand-assembled notebook)

One 4-bit base model is loaded ONCE. The adapter is a small add-on that is switched on
and off, and "baseline" simply runs with the adapter disabled. This keeps VRAM use low and
makes switching instant.

Both modes run through our own `AssembledVLM` (vision encoder -> projector -> splice ->
M-RoPE -> language model -> lm_head, with a hand-written KV-cached greedy decoder). It shares
its modules with the Hugging Face model, so the LoRA layers (and their on/off switch) apply
to it automatically.

IMPORTANT: `schema_dict` and `instruction` must stay identical to the training prompt.
"""
import contextlib
import inspect
import json
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image

BASE_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'
MAX_SIDE = 800
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.0     # 1.0 = pure greedy decoding
MODE_LABELS = {
    'baseline': 'Baseline (no fine-tuning)',
    'finetuned': 'Fine-tuned',
}

# ---------------- Prompt (must match training) ----------------
# ---------------- Schema and prompt ----------------
schema_dict = {
    'menu': {
        'nm': 'name of the menu', 'num': 'identification number of menu',
        'unitprice': 'unit price of menu', 'cnt': 'quantity of menu',
        'discountprice': 'discounted price of menu', 'price': 'total price of menu',
        'itemsubtotal': 'price of each menu after discount applied',
        'vatyn': 'whether the price includes tax or not', 'etc': 'others',
        'sub': {'nm': 'name of submenu', 'unitprice': 'unit price of submenu',
                'cnt': 'quantity of submenu', 'price': 'total price of submenu', 'etc': 'others'}
    },
    'sub_total': {
        'price': 'subtotal price', 'discount_price': 'discounted price in total',
        'service_price': 'service charge', 'othersvc_price': 'added charge other than service charge',
        'tax_price': 'tax amount', 'etc': 'others'
    },
    'total': {
        'total_price': 'total price', 'etc': 'others', 'cashprice': 'amount of price paid in cash',
        'changeprice': 'amount of change in cash',
        'creditcardprice': 'amount of price paid in credit/debit card',
        'emoneyprice': 'amount of price paid in emoney, point',
        'menutype_cnt': 'total count of type of menu', 'menuqty_cnt': 'total count of quantity'
    }
}

instruction = (
    'You are an expert at reading receipts and invoices. '
    'Extract the information from the receipt image into valid JSON that follows this schema. '
    'Copy text and numbers exactly as they are printed. '
    'Leave a key out if the information is not visible. '
    'Return only the JSON, with no explanation. Schema: '
    + json.dumps(schema_dict)
)
print(instruction[:300], '...')

# ---------------- OpenCV preprocessing (same as training) ----------------
def preprocess(pil_img, max_side=MAX_SIDE):
    img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------- Output parsing and a simple accuracy metric ----------------
def parse_json(text):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def flatten_dict(d, parent_key=''):
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}.{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            if v and all(isinstance(i, dict) for i in v):
                for idx, item in enumerate(v):
                    items.update(flatten_dict(item, f'{new_key}[{idx}]'))
            else:
                items[new_key] = v
        else:
            items[new_key] = v
    return items


def normalize(d):
    if not isinstance(d, dict):
        return {}
    d = dict(d)
    if isinstance(d.get('menu'), dict):
        d['menu'] = [d['menu']]
    return d


def field_accuracy(gt, pred):
    """Fraction of ground-truth fields that the prediction reproduces exactly."""
    gt_flat = flatten_dict(normalize(gt))
    if pred is None or not gt_flat:
        return 0.0
    pred_flat = flatten_dict(normalize(pred))
    return sum(1 for k, v in gt_flat.items() if pred_flat.get(k, '') == v) / len(gt_flat)


# ---------------- Our own model class (same as the assembly notebook) ----------------
import torch


class AssembledVLM(torch.nn.Module):
    """Qwen2.5-VL wired by hand from its parts. Shares the modules with the Hugging Face model."""

    def __init__(self, visual, text_model, lm_head, embed_tokens, rope_fn, image_token_id):
        super().__init__()
        self.visual, self.text_model, self.lm_head = visual, text_model, lm_head
        self.embed_tokens, self.rope_fn, self.image_token_id = embed_tokens, rope_fn, image_token_id

    def _image_embeds(self, pixel_values, image_grid_thw):
        dtype = next(self.visual.parameters()).dtype
        out = self.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
        if not isinstance(out, torch.Tensor):               # newer versions return an output object
            pooled = getattr(out, 'pooler_output', None)
            out = pooled if pooled is not None else out[0]
        return out

    def _inputs_embeds(self, input_ids, pixel_values, image_grid_thw):
        image_embeds = self._image_embeds(pixel_values, image_grid_thw)      # 1) vision + projector
        emb = self.embed_tokens(input_ids)                                   # 2) text embeddings
        mask = input_ids == self.image_token_id
        if int(mask.sum()) != image_embeds.shape[0]:
            raise ValueError(f'{int(mask.sum())} placeholder tokens but {image_embeds.shape[0]} image vectors')
        return emb.masked_scatter(mask.unsqueeze(-1).expand_as(emb), image_embeds.to(emb.dtype))   # 3) splice

    def _positions(self, input_ids, attention_mask, image_grid_thw, mm_token_type_ids=None):
        # 4) multimodal (3-D) rotary position ids
        kwargs = dict(input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        if 'mm_token_type_ids' in inspect.signature(self.rope_fn).parameters:   # newer transformers
            if mm_token_type_ids is None:                      # 0 = text, 1 = image token
                mm_token_type_ids = (input_ids == self.image_token_id).long()
            kwargs['mm_token_type_ids'] = mm_token_type_ids
        return self.rope_fn(**kwargs)                          # (position_ids, rope_deltas)

    def _hidden(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None):
        emb = self._inputs_embeds(input_ids, pixel_values, image_grid_thw)
        position_ids, _ = self._positions(input_ids, attention_mask, image_grid_thw, mm_token_type_ids)
        out = self.text_model(input_ids=None, inputs_embeds=emb, position_ids=position_ids,   # 5) decoder
                              attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]

    def logits(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None, last_n=None):
        h = self._hidden(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
        if last_n: h = h[:, -last_n:]
        return self.lm_head(h)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels, mm_token_type_ids=None):
        h = self._hidden(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
        shift_h, shift_labels = h[:, :-1], labels[:, 1:]       # position t predicts token t+1
        sel = shift_labels != -100                              # answer tokens only
        return torch.nn.functional.cross_entropy(self.lm_head(shift_h[sel]).float(), shift_labels[sel])

    def _pick(self, h, seen_ids, repetition_penalty=1.0):
        logits = self.lm_head(h[:, -1:])[:, 0]                          # (1, vocab)
        if repetition_penalty != 1.0:
            score = logits.gather(1, seen_ids)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits = logits.scatter(1, seen_ids, score)
        return logits.argmax(-1, keepdim=True)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None,
                 max_new_tokens=1024, eos_token_ids=(), repetition_penalty=1.0):
        """Greedy decoding with a KV cache. Prefill the prompt once, then one token per step
        (its M-RoPE position is sequence index + rope_deltas, the same on all 3 axes)."""
        from transformers import DynamicCache
        emb = self._inputs_embeds(input_ids, pixel_values, image_grid_thw)
        position_ids, rope_deltas = self._positions(input_ids, attention_mask, image_grid_thw, mm_token_type_ids)
        cache = DynamicCache()
        out = self.text_model(input_ids=None, inputs_embeds=emb, position_ids=position_ids,
                              attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        next_tok = self._pick(h, input_ids, repetition_penalty)
        generated, seq_len = [next_tok], input_ids.shape[1]
        for _ in range(max_new_tokens - 1):
            if int(next_tok) in eos_token_ids:
                break
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=1)
            pos = (torch.full((1, 1), seq_len, device=input_ids.device) + rope_deltas).long()
            out = self.text_model(input_ids=None, inputs_embeds=self.embed_tokens(next_tok),
                                  position_ids=pos.unsqueeze(0).expand(3, -1, -1),
                                  attention_mask=attention_mask, past_key_values=cache, use_cache=True)
            h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
            next_tok = self._pick(h, torch.cat([input_ids] + generated, dim=1), repetition_penalty)
            generated.append(next_tok)
            seq_len += 1
        return torch.cat(generated, dim=1)                               # new tokens only


def assemble(hf):
    """Find the parts of a Hugging Face Qwen2.5-VL model and wire them into an AssembledVLM."""
    inner = hf.model
    visual = getattr(hf, 'visual', None)
    if visual is None:
        visual = inner.visual
    text_model = getattr(inner, 'language_model', None)
    if text_model is None:
        text_model = inner                                  # older transformers
    rope_fn = getattr(hf, 'get_rope_index', None) or getattr(inner, 'get_rope_index')
    return AssembledVLM(visual, text_model, hf.lm_head, hf.get_input_embeddings(), rope_fn,
                        hf.config.image_token_id)


# ---------------- The model engine ----------------
class Engine:
    def __init__(self, adapters=None, base_model=BASE_MODEL):
        """adapters: {'finetuned': folder}. A missing folder is skipped (baseline only)."""
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.lock = threading.Lock()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['visual', 'merger', 'lm_head'])   # vision + projector stay fp16
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, quantization_config=bnb_config, torch_dtype=torch.float16,
            device_map={'': 0}, attn_implementation='sdpa')
        self.processor = AutoProcessor.from_pretrained(base_model)

        found = {}
        for name, path in (adapters or {}).items():
            if path and os.path.isdir(path):
                found[name] = path
            else:
                print(f'[info] adapter "{name}" not found at {path!r}, that mode is skipped')
        self.has_adapters = bool(found)
        if found:
            from peft import PeftModel
            names = list(found)
            self.model = PeftModel.from_pretrained(base, found[names[0]], adapter_name=names[0])
            for n in names[1:]:
                self.model.load_adapter(found[n], adapter_name=n)
            hf = self.model.get_base_model()          # the Hugging Face model with LoRA layers injected
        else:
            self.model = hf = base
        self.model.eval()
        self.modes = ['baseline'] + list(found)

        # Our own model, built from the SAME modules (no copies): LoRA layers and the
        # adapter on/off switch therefore apply to it too.
        self.assembled = assemble(hf)
        eos = hf.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

        # The projector ("merger") is shared by all modes. If an adapter folder contains a
        # trained projector (projector.pt), it is swapped in only for that mode.
        self._merger = self.assembled.visual.merger
        self._merger_orig = {k: v.detach().clone() for k, v in self._merger.state_dict().items()}
        self._merger_custom = {}
        for n, d in found.items():
            f = os.path.join(d, 'projector.pt')
            if os.path.isfile(f):
                self._merger_custom[n] = torch.load(f, map_location='cuda')
                print(f'[info] mode "{n}" has its own trained projector')
        self._merger_current = None          # None = original projector is loaded

    # --- switching modes ---
    def _load_projector(self, state):
        key = id(state) if state is not None else None
        if key == self._merger_current:
            return
        self._merger.load_state_dict(state if state is not None else self._merger_orig)
        self._merger_current = key

    @contextlib.contextmanager
    def _use(self, mode):
        if mode not in self.modes:
            raise ValueError(f'unknown mode {mode!r}; available: {self.modes}')
        self._load_projector(None if mode == 'baseline' else self._merger_custom.get(mode))
        if mode == 'baseline':
            ctx = self.model.disable_adapter() if self.has_adapters else contextlib.nullcontext()
        else:
            self.model.set_adapter(mode)
            ctx = contextlib.nullcontext()
        with ctx:
            yield

    # --- inputs ---
    def _user_msg(self):
        return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': instruction}]}

    _KEYS = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids')

    def _prompt_inputs(self, pil_img):
        text = self.processor.apply_chat_template([self._user_msg()], tokenize=False,
                                                  add_generation_prompt=True)
        enc = self.processor(text=[text], images=[preprocess(pil_img)], return_tensors='pt')
        return {k: enc[k].to('cuda') for k in self._KEYS if k in enc}

    def _training_inputs(self, pil_img, gt_parse):
        p = self.processor
        img = preprocess(pil_img)
        user = self._user_msg()
        asst = {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(gt_parse)}]}
        full_text = p.apply_chat_template([user, asst], tokenize=False, add_generation_prompt=False)
        prompt_text = p.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full = p(text=[full_text], images=[img], return_tensors='pt')
        prompt_len = p(text=[prompt_text], images=[img], return_tensors='pt')['input_ids'].shape[1]
        labels = full['input_ids'].clone()
        labels[:, :prompt_len] = -100
        inputs = {k: full[k].to('cuda') for k in self._KEYS if k in full}
        return inputs, labels.to('cuda')

    # --- inference (our own AssembledVLM.generate) ---
    def extract(self, pil_img, mode):
        """Returns (parsed dict or None, raw model text, seconds)."""
        torch = self.torch
        inputs = self._prompt_inputs(pil_img)
        t0 = time.time()
        with self.lock:
            with self._use(mode), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                new = self.assembled.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                              eos_token_ids=self.eos_ids,
                                              repetition_penalty=REPETITION_PENALTY)[0]
        raw = self.processor.tokenizer.decode(new, skip_special_tokens=True)
        return parse_json(raw), raw, time.time() - t0

    # --- checks ---
    def example_loss(self, mode, pil_img, gt_parse):
        """Training-style loss of one receipt under a mode, computed by AssembledVLM
        (used to check that the adapter is really active)."""
        torch = self.torch
        inputs, labels = self._training_inputs(pil_img, gt_parse)
        with self.lock:
            with self._use(mode), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                return self.assembled(**inputs, labels=labels).item()

    def compare_with_library(self, mode, pil_img, gt_parse, last_n=50):
        """Our logits vs the Hugging Face model's logits for the same input and mode."""
        torch = self.torch
        inputs, _ = self._training_inputs(pil_img, gt_parse)
        with self.lock:
            with self._use(mode), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                mine = self.assembled.logits(**inputs, last_n=last_n).float()
                ref = self.model(**inputs).logits[:, -last_n:].float()
        return (mine - ref).abs().max().item(), (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()


# ---------------- User interface (Gradio) ----------------
def build_ui(engine):
    import gradio as gr

    choices = [(MODE_LABELS[m], m) for m in engine.modes]
    if len(engine.modes) > 1:
        choices.append(('Compare both models', 'all'))
    default = 'finetuned' if 'finetuned' in engine.modes else 'baseline'

    def ui_fn(img, choice):
        if img is None:
            return None, '', ''
        modes = engine.modes if choice == 'all' else [choice]
        results, raws, timing = {}, [], []
        for m in modes:
            data, raw, secs = engine.extract(img, m)
            results[MODE_LABELS[m]] = data if data is not None else {'error': 'output was not valid JSON'}
            raws.append(f'=== {MODE_LABELS[m]} ===\n{raw}')
            timing.append(f'{MODE_LABELS[m]}: {secs:.1f} s')
        shown = results[MODE_LABELS[modes[0]]] if len(modes) == 1 else results
        return shown, '\n\n'.join(raws), ' | '.join(timing)

    with gr.Blocks(title='Receipt extractor') as demo:
        gr.Markdown('# Receipt extractor\nUpload or capture a receipt photo, choose a model, and get structured JSON.')
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type='pil', sources=['upload', 'webcam'], label='Receipt photo')
                choice = gr.Radio(choices=choices, value=default, label='Model')
                btn = gr.Button('Extract', variant='primary')
            with gr.Column():
                out_json = gr.JSON(label='Extracted data')
                out_time = gr.Textbox(label='Time', lines=1)
                out_raw = gr.Textbox(label='Raw model output', lines=8)
        btn.click(ui_fn, [inp, choice], [out_json, out_raw, out_time])
    return demo


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--finetuned', default='lora_receipt_3b_trl')
    ap.add_argument('--share', action='store_true')
    a = ap.parse_args()
    build_ui(Engine({'finetuned': a.finetuned})).launch(share=a.share)