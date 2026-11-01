# Receipt Information Extraction with Hand-Assembled Qwen2.5-VL & TRL Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow.svg)](https://huggingface.co/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--3B--Instruct-FF6F00.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-CORD--v2-green.svg)](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20Demo-orange.svg)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Platform-Google%20Colab%20(NVIDIA%20T4)-76B900.svg)](https://colab.research.google.com/)

An end-to-end cloud-based visual receipt information extraction system built with **Qwen2.5-VL-3B**, developed for the **Trends in Computer Vision (*Tren pada Visi Komputer*)** graduate exploration.

This project goes beyond standard black-box library pipelines by **dismantling Qwen2.5-VL into its atomic sub-components**, re-wiring the multimodal forward pass and autoregressive KV-cached decoder by hand (`AssembledVLM`), proving mathematical and token-level equivalence with Hugging Face reference implementations, fine-tuning via **TRL (`SFTTrainer`)** and **QLoRA** with custom NaN protection, and serving both baseline and fine-tuned models in an interactive single-GPU **Gradio web application**.

---

## 📌 Table of Contents

- [Overview & Key Innovations](#-overview--key-innovations)
- [System Architecture](#-system-architecture)
- [Dataset & Schema Specification](#-dataset--schema-specification)
- [Methodology & Implementation Details](#-methodology--implementation-details)
  - [1. OpenCV Aspect-Ratio Preserving Preprocessing](#1-opencv-aspect-ratio-preserving-preprocessing)
  - [2. Hand-Assembled VLM Architecture (`AssembledVLM`)](#2-hand-assembled-vlm-architecture-assembledvlm)
  - [3. Equivalence & Parity Verification](#3-equivalence--parity-verification)
  - [4. Parameter-Efficient Fine-Tuning (QLoRA + TRL)](#4-parameter-efficient-fine-tuning-qlora--trl)
  - [5. NaN-Safe Loss Protection for T4 GPUs](#5-nan-safe-loss-protection-for-t4-gpus)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
- [Interactive Gradio Application](#-interactive-gradio-application)
- [Repository Structure](#-repository-structure)
- [Quickstart & Usage](#-quickstart--usage)
  - [Prerequisites & Environment](#prerequisites--environment)
  - [Running the Training Notebook](#running-the-training-notebook)
  - [Launching the Web Application](#launching-the-web-application)
- [Project Metadata & Acknowledgments](#-project-metadata--acknowledgments)

---

## 💡 Overview & Key Innovations

Visual Document Understanding (VDU) on real-world retail receipts poses unique challenges: arbitrary camera orientations, vertical aspect ratios, faded thermal ink, irregular fonts, and dense multi-line key-value pairs. 

This project demonstrates how modern cloud-native tools on the **Hugging Face Hub** and **Google Colab** can be combined with deep structural engineering to achieve state-of-the-art information extraction on consumer hardware (16 GB NVIDIA T4):

1. **Complete Architectural Transparency**: Rather than treating `Qwen2_5_VLForConditionalGeneration` as an opaque black box, the model is decomposed into its **Vision Transformer (ViT)**, **Projector Merger (`visual.merger`)**, **Word Embeddings**, and **Language Model Decoder**. The forward pass and KV-cached greedy generation are re-implemented from scratch.
2. **Proven Equivalence**: Rigorous mathematical parity checks verify that our custom `AssembledVLM` produces identical logits ($\Delta < 10^{-3}$, $> 99.9\%$ argmax agreement), identical generated tokens, and matching LoRA backward gradient flow compared to Hugging Face reference code.
3. **Robust SFT Training with TRL**: Custom collator masks prompt tokens (`labels = -100`), ensuring loss gradients update exclusively on structured JSON targets. Subclassed `NaNSafeSFTTrainer` prevents fp16 gradient explosion and silent weight corruption on noisy receipt samples.
4. **Doubled Extraction Performance**: QLoRA adaptation boosts exact hierarchical field accuracy on unseen CORD-v2 test receipts from **40.69% to 80.94%**, while increasing total price matching to **89.47%**.
5. **Zero-Redundancy Dual-Model Serving**: Baseline and fine-tuned models are served concurrently on a single 16 GB T4 GPU using dynamic adapter switching (`disable_adapter()` vs `set_adapter('finetuned')`), eliminating the need to duplicate base model weights in VRAM.

---

## 🏗️ System Architecture

```
                                [ Input Receipt Image ]
                                           │
                                           ▼
                       [ OpenCV Preprocessing (Aspect-Ratio Safe) ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [ Vision Encoder (ViT) ]                       [ User Instructions ]
          - Native resolution patches                    - Hierarchical JSON schema
          - FP16 precision                                        │
                    │                                             ▼
                    ▼                                   [ Tokenizer & Embeddings ]
        [ Projector (visual.merger) ]                    - Input token IDs
        - 2x2 patch spatial merging                      - <|image_pad|> placeholders
        - Dimension projection                                    │
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                        [ Multimodal Tensor Splicing ]
                        - masked_scatter: image tokens overwrite <|image_pad|>
                                           │
                                           ▼
                        [ 3D Multimodal RoPE (M-RoPE) ]
                        - Temporal, Height, and Width coordinates
                                           │
                                           ▼
                     [ 4-bit Quantized Language Model (NF4) ]
                     - Qwen2.5-VL-3B Transformer Backbone
                     - LoRA Adapters (q, k, v, o, gate, up, down)
                                           │
                                           ▼
                       [ AssembledVLM Autoregressive Head ]
                       - KV-cached greedy token generation
                                           │
                                           ▼
                       [ Formatted Hierarchical JSON Parse ]
```

---

## 📊 Dataset & Schema Specification

The model is trained and evaluated on the benchmark dataset **[naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)** (Consolidated Receipt Dataset for Post-OCR Parsing):
- **Train split**: 800 receipts
- **Validation split**: 100 receipts
- **Test split**: 100 receipts

### Ground Truth Schema

Extraction targets strict hierarchical JSON with three primary sections:

```json
{
  "menu": [
    {
      "nm": "name of the menu item",
      "num": "menu identifier code",
      "unitprice": "unit price",
      "cnt": "quantity purchased",
      "discountprice": "applied item discount",
      "price": "total line-item price",
      "itemsubtotal": "net price after discount",
      "vatyn": "tax inclusion indicator",
      "etc": "auxiliary line remarks",
      "sub": {
        "nm": "submenu / modifier name",
        "unitprice": "submenu unit price",
        "cnt": "submenu quantity",
        "price": "submenu total price",
        "etc": "modifier remarks"
      }
    }
  ],
  "sub_total": {
    "price": "subtotal before tax/fees",
    "discount_price": "aggregate discount amount",
    "service_price": "service charge",
    "othersvc_price": "miscellaneous charges",
    "tax_price": "total tax / VAT amount",
    "etc": "other subtotal notes"
  },
  "total": {
    "total_price": "grand total payable",
    "cashprice": "cash tendered",
    "changeprice": "change returned",
    "creditcardprice": "card payment amount",
    "emoneyprice": "digital wallet / points payment",
    "menutype_cnt": "count of distinct items",
    "menuqty_cnt": "aggregate count of units",
    "etc": "miscellaneous total notes"
  }
}
```

---

## 🔬 Methodology & Implementation Details

### 1. OpenCV Aspect-Ratio Preserving Preprocessing

Receipts are naturally tall, narrow strips of paper. Standard square resizing distortions compress text horizontally, rendering small point numbers unreadable.

We employ an aspect-ratio-safe OpenCV pipeline:
```python
def preprocess(pil_img, max_side=800):
    img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```
- **Area Interpolation (`cv2.INTER_AREA`)**: Optimal downsampling algorithm for high-frequency alphanumeric text.
- **Uniform Scaling**: Retains true typographic aspect ratios while limiting maximum pixel dimensions to constrain GPU VRAM usage.

### 2. Hand-Assembled VLM Architecture (`AssembledVLM`)

In `receipt_assembly_trl_qwen25vl_colab.ipynb`, we bypass high-level wrapper wrappers and wire the submodules manually:

```python
class AssembledVLM(torch.nn.Module):
    def __init__(self, visual, text_model, lm_head, embed_tokens, rope_fn, image_token_id):
        super().__init__()
        self.visual, self.text_model, self.lm_head = visual, text_model, lm_head
        self.embed_tokens, self.rope_fn, self.image_token_id = embed_tokens, rope_fn, image_token_id
```

The forward execution pipeline executes five precise steps:
1. **Vision Encoding & Projection**: `visual(pixel_values, grid_thw=image_grid_thw)` processes patch tokens through spatial merger layers into language-aligned visual embeddings.
2. **Token Embedding**: Instruction text is projected into vector space via `embed_tokens(input_ids)`.
3. **Multimodal Splicing**: Tensor mask `input_ids == image_token_id` identifies placeholder positions, and `masked_scatter` injects projected image vectors directly into the sequence embedding.
4. **3D M-RoPE Positional Indexing**: Rotary positional coordinates across 3 dimensions (grid time, height, and width) are generated via `rope_fn`.
5. **Autoregressive Decoding**: Text decoder evaluates the mixed multimodal sequence. For generation, a custom hand-written KV cache buffers prior keys and values, feeding exactly one token per step with proper position offsets.

### 3. Equivalence & Parity Verification

To ensure zero divergence between our manual assembly and the Hugging Face reference codebase, three verification benchmarks are executed prior to training:

- **Logit Match**: Maximum absolute difference between `assembled.logits` and `hf_model.logits` is verified to be under $0.001$, with $> 99.9\%$ argmax token prediction overlap.
- **Autoregressive Generation Parity**: Greedy generation outputs from `AssembledVLM.generate` are compared token-by-token against `hf_model.generate` across test receipts.
- **Gradient Flow Verification**: Backpropagation through `AssembledVLM` confirms all active LoRA parameters receive finite, non-zero gradients before training starts.

### 4. Parameter-Efficient Fine-Tuning (QLoRA + TRL)

- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct` loaded in 4-bit NormalFloat (NF4) with double quantization.
- **Unquantized Modules**: `visual`, `merger`, and `lm_head` remain in FP16 precision via `llm_int8_skip_modules`.
- **LoRA Configuration**:
  - Rank ($r$): 16, Alpha ($\alpha$): 16, Dropout: 0.0
  - Target modules: Language decoder projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - Total model parameters: ~3.78 Billion.
  - Trainable parameters: **29,933,568 (~0.79% of total)**.
- **Loss Masking**: Custom data collator assigns `labels = -100` to all prompt tokens, ensuring backpropagation operates solely on the assistant's JSON response tokens.

### 5. NaN-Safe Loss Protection for T4 GPUs

NVIDIA T4 GPUs lack hardware support for `bfloat16`, necessitating `float16` training. On long multimodal sequences, FP16 gradients can occasionally suffer underflow or overflow, yielding `NaN` loss values that ruin model weights.

We subclass TRL's `SFTTrainer` with an active numerical circuit breaker:

```python
class NaNSafeSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = self.assembled_model(**inputs)
        if torch.isfinite(loss):
            self.consecutive_bad = 0
            return loss
        else:
            self.skipped += 1
            self.consecutive_bad += 1
            # Return a zero loss that retains autograd connectivity without altering weights
            dummy_loss = 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)
            return dummy_loss
```

---

## 📈 Experimental Results & Benchmarks

Evaluation was conducted over the complete 100-receipt CORD-v2 test set (`results_trl.csv`):

| Evaluation Metric | Qwen2.5-VL-3B 4-bit Baseline | Qwen2.5-VL-3B QLoRA (AssembledVLM + TRL) | Absolute Delta |
|:---|:---:|:---:|:---:|
| **Hierarchical Field Accuracy** | **40.69%** | **80.94%** | **+40.25% (2x improvement)** |
| **Total Price Exact Match** | **78.95%** | **89.47%** | **+10.52%** |
| **JSON Parse Failures** | 6 / 100 | **2 / 100** | **-66.7% failure reduction** |
| **Mean Latency per Receipt** | 31.69 s | **23.56 s** | **-8.13 s faster** |
| **Peak VRAM (Inference)** | 2.79 GB | 3.28 GB | Fits easily on 16GB T4 |
| **Peak VRAM (Training)** | — | **4.74 GB** | Low memory footprint |

> **Key Observation**: Fine-tuning not only doubled the key-value field extraction accuracy, but also reduced average inference latency from 31.69s to 23.56s. The fine-tuned model immediately outputs compact, compliant JSON tokens without hallucinating boilerplate conversational commentary.

### Qualitative Prediction Sample

```
================================== GROUND TRUTH ==================================
{
  "menu": [
    {"nm": "J.STB PROMO", "price": "17500"},
    {"nm": "Y.B.BAT", "price": "46000"},
    {"nm": "Y.BASO PROM", "price": "27500"}
  ],
  "total": {
    "total_price": "91000",
    "cashprice": "91000"
  }
}

============================== FINE-TUNED PREDICTION ==============================
{
  "menu": [
    {"nm": "J.STB PROMO", "price": "17500"},
    {"nm": "Y.B.BAT", "price": "46000"},
    {"nm": "Y.BASO PROM", "price": "27500"}
  ],
  "total": {
    "total_price": "91000",
    "cashprice": "91000"
  }
}
```

---

## 🖥️ Interactive Gradio Application

The web interface in `receipt_app_colab_trl_assembled.ipynb` provides an interactive testing workbench for comparing baseline and fine-tuned models side by side.

### Single-GPU Memory Optimization
Running two 3B models simultaneously would typically exceed T4 memory limits. We circumvent this with zero-copy adapter switching:
- The base 4-bit Qwen2.5-VL model is loaded **once** in VRAM.
- `AssembledVLM` wraps the active base modules.
- **Baseline execution**: Calls `model.disable_adapter()` (pure base weights).
- **Fine-tuned execution**: Calls `model.set_adapter('finetuned')` (activates LoRA weights).
- Mode switching is **instantaneous** and requires **0 additional base model memory**.

### Web App Capabilities
- **Flexible Image Input**: Direct image upload or camera snapshot via webcam.
- **Model Selector**: Switch between `Baseline`, `Fine-tuned`, or `Compare both models`.
- **Side-by-Side Comparison**: Concurrently displays parsed JSON trees, raw generation text, and execution latency.
- **Public URL**: Shareable via Gradio live link for testing on mobile smartphones.

---

## 📂 Repository Structure

```
├── Cloud-based Exploration on the Hugging Face platform for Receipt Extraction Task.pdf   # Slide deck report (PDF)
├── Cloud-based Exploration on the Hugging Face platform for Receipt Extraction Task.pptx  # Slide deck presentation (PPTX)
├── receipt_assembly_trl_qwen25vl_colab.ipynb                                              # Core training & assembly notebook
├── receipt_app_colab_trl_assembled.ipynb                                                  # Web application deployment notebook
├── receipt_app_multi.py                                                                   # Standalone engine & Gradio app script
├── results_trl.csv                                                                        # Empirical evaluation results
├── .gitignore                                                                             # Git ignore rules (ignores *.zip, caches)
└── README.md                                                                              # Project documentation
```

> [!NOTE]
> Large model weight archives (such as `lora_receipt_3b_trl.zip` and the `lora_receipt_3b_trl/` directory, ~108 MB) are excluded from version control via `.gitignore` to keep the repository lightweight and below GitHub's standard file limit. The fine-tuned LoRA adapter can be trained directly in the training notebook or downloaded via GitHub Releases / Hugging Face Hub.

### Notebook Functionality

1. **`receipt_assembly_trl_qwen25vl_colab.ipynb`**:
   - Environment setup and Hugging Face library installation.
   - CORD-v2 dataset download and inspection.
   - Decomposition of `Qwen2_5_VLForConditionalGeneration` into sub-modules.
   - Re-implementation of `AssembledVLM` with custom KV-cached `generate()`.
   - Logit parity, token equivalence, and gradient verification tests.
   - Zero-shot baseline evaluation across 100 test receipts.
   - QLoRA setup, custom data collator, and `NaNSafeSFTTrainer` execution.
   - Post-training evaluation and saving the fine-tuned adapter (`lora_receipt_3b_trl/`) and benchmark metrics.

2. **`receipt_app_colab_trl_assembled.ipynb`**:
   - Setup and environment preparation.
   - Adapter loading (reads `lora_receipt_3b_trl/` directory or unzips uploaded adapter).
   - Generation of modular `receipt_app_multi.py`.
   - Instantiation of `Engine` with single-GPU dual-mode dispatch.
   - In-app adapter validation (loss difference test on sample receipts).
   - Interactive Gradio Blocks UI launch with public tunnel URL.

---

## 🚀 Quickstart & Usage

### Prerequisites & Environment

- **Compute**: Google Colab with NVIDIA T4 GPU (or any GPU with $\ge 12$ GB VRAM).
- **Python**: 3.10+
- **Core Dependencies**:
  ```bash
  pip install -q -U transformers peft trl bitsandbytes accelerate gradio opencv-python-headless datasets
  ```

### Running the Training Notebook

1. Open `receipt_assembly_trl_qwen25vl_colab.ipynb` in Google Colab.
2. Under **Runtime** > **Change runtime type**, select **T4 GPU**.
3. Run all cells sequentially.
4. The notebook will:
   - Verify GPU capability and memory.
   - Assemble `AssembledVLM` and validate parity against Hugging Face reference.
   - Evaluate baseline performance on 100 test receipts.
   - Train the LoRA adapter for 2 epochs using `NaNSafeSFTTrainer`.
   - Evaluate fine-tuned performance, save the adapter folder (`lora_receipt_3b_trl/`), and output `results_trl.csv`.

### Launching the Web Application

#### Option A: In Google Colab
1. Open `receipt_app_colab_trl_assembled.ipynb`.
2. Ensure the `lora_receipt_3b_trl/` adapter directory is present (or upload your trained adapter zip when prompted).
3. Execute the notebook cells. The final cell will output a public Gradio URL:
   ```
   Running on public URL: https://xxxxxxxx.gradio.live
   ```

#### Option B: Standalone CLI
If running on a local machine with an NVIDIA GPU:
```bash
# Point directly to your fine-tuned adapter folder
python receipt_app_multi.py --finetuned lora_receipt_3b_trl --share
```
*(If you have a downloaded `lora_receipt_3b_trl.zip`, unzip it first: `unzip lora_receipt_3b_trl.zip -d lora_receipt_3b_trl`)*

---

## 👥 Project Metadata & Acknowledgments

- **Author**: Maritza Amalia Dwiputri
- **Student ID (NIM)**: 203012610017
- **Course**: *Tren pada Visi Komputer* (Trends in Computer Vision) — Master's Degree (S2) Informatics / Computer Science
- **Ecosystem**:
  - [Hugging Face](https://huggingface.co/) for `transformers`, `datasets`, `peft`, `trl`, and `accelerate`.
  - [Qwen Team / Alibaba Cloud](https://github.com/QwenLM/Qwen2.5-VL) for the `Qwen2.5-VL-3B-Instruct` multimodal foundation model.
  - [NAVER Clova AI](https://github.com/clovaai/cord) for the `CORD-v2` benchmark dataset.
  - [Gradio](https://gradio.app/) for the interactive web deployment framework.
