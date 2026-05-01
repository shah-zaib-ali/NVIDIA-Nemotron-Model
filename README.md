<div align="center">

# 🧠 NVIDIA Nemotron Reasoning Challenge

**Fine-tuned LLM submission for the Kaggle NVIDIA Nemotron-3 Nano Reasoning Competition**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NVIDIA%20Nemotron%20Reasoning-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
[![Model](https://img.shields.io/badge/Model-Nemotron--3--Nano--30B-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://huggingface.co/nvidia)
[![Framework](https://img.shields.io/badge/Inference-vLLM-FF6B35?style=for-the-badge&logo=python&logoColor=white)](https://github.com/vllm-project/vllm)
[![Adapter](https://img.shields.io/badge/Adapter-LoRA-9B59B6?style=for-the-badge&logo=pytorch&logoColor=white)](https://huggingface.co/docs/peft/conceptual_guides/lora)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

*A high-performance reasoning pipeline using NVIDIA's Nemotron-3 Nano 30B model with LoRA adapters, multi-sample voting, and category-aware prompt engineering to solve symbolic and mathematical reasoning puzzles.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Problem Categories](#-problem-categories)
- [Pipeline Flow](#-pipeline-flow)
- [Prompt Engineering Strategy](#-prompt-engineering-strategy)
- [Multi-Sample Voting](#-multi-sample-voting)
- [Answer Extraction](#-answer-extraction)
- [Configuration](#%EF%B8%8F-configuration)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)

---

## 🔍 Overview

This repository contains the inference and evaluation notebook for the **[NVIDIA Nemotron-3 Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)** on Kaggle. The challenge tasks participants with solving symbolic reasoning puzzles across 6 distinct categories — from bit manipulation and cryptarithmetic to unit conversion and secret equation rules.

The solution leverages:

- **Nemotron-3-Nano-30B** as the base reasoning model (30B parameters, BF16 precision)
- **LoRA fine-tuned adapters** for task-specific alignment
- **vLLM** for high-throughput batched inference
- **Category-aware prompt engineering** to tailor system instructions per puzzle type
- **Multi-sample majority voting** (5 samples with dynamic temperature) for robust answer selection

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Inference Pipeline                               │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐    │
│  │  Input CSV   │────▶│  Category    │────▶│   Prompt Builder     │    │
│  │  (prompts)   │     │  Detector    │     │  (per-category hints)│    │
│  └──────────────┘     └──────────────┘     └──────────┬───────────┘    │
│                                                         │               │
│                                                         ▼               │
│                                           ┌─────────────────────────┐  │
│                                           │      vLLM Engine        │  │
│                                           │   Nemotron-3-Nano-30B   │  │
│                                           │   + LoRA Adapter        │  │
│                                           │   enable_thinking=True  │  │
│                                           └──────────┬──────────────┘  │
│                                                      │ × 5 samples     │
│                                                      ▼               │  │
│                                           ┌─────────────────────────┐  │
│                                           │  Answer Extractor       │  │
│                                           │  \boxed{} → regex       │  │
│                                           └──────────┬──────────────┘  │
│                                                      │               │  │
│                                                      ▼               │  │
│                                           ┌─────────────────────────┐  │
│                                           │  Majority Voter         │  │
│                                           │  (category-aware)       │  │
│                                           └──────────┬──────────────┘  │
│                                                      │                  │
│                                                      ▼                  │
│                                           ┌─────────────────────────┐  │
│                                           │   submission.zip        │  │
│                                           └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Problem Categories

The competition features **6 distinct reasoning puzzle categories**, each requiring unique reasoning strategies. The pipeline detects each category automatically from the prompt content:

| Category | Detection Signal | Reasoning Strategy | Voting Mode |
|---|---|---|---|
| `bit_manipulation` | "secret bit manipulation rule transforms 8-bit binary numbers" | Binary-aware; no leading zero padding | Majority vote |
| `cipher` | "secret encryption rules are used on text" | Pattern mapping; character substitution | Majority vote |
| `equation_numeric_deduce` | Numeric examples in equations | Rule induction from examples; symbols allowed | Majority vote |
| `equation_numeric_guess` | Non-numeric equation hints | Symbolic rule inference; no pure digit assumption | Majority vote |
| `cryptarithm_deduce` / `_guess` | No digits in examples | Letter-to-digit mapping; multiple pattern tries | Majority vote |
| `gravity` | "gravitational constant has been secretly changed" | Numeric physics reasoning | Force numeric vote |
| `numeral` | "converted into a different numeral system" | Base conversion logic | Force numeric vote |
| `unit_conversion` | "secret unit conversion is applied to measurements" | Unit ratio inference | Force numeric vote |

### Category Distribution

```
bit_manipulation   ████████████████████  ~17%
cipher             ███████████████████   ~16%
equation_numeric   ████████████████████  ~17%
cryptarithm        ████████████████████  ~17%
gravity            ██████████████        ~11%
numeral            ██████████████        ~11%
unit_conversion    ███████████████       ~12%
```

---

## 🔄 Pipeline Flow

```
1. SETUP
   └── Load submission.zip (LoRA adapter weights)
   └── Load adapter_config.json
   └── Download Nemotron-3-Nano-30B from KaggleHub

2. MODEL CACHE WARMUP
   └── cache_model() — parallel pre-read with 16 workers, 1GB chunks
   └── Warms OS page cache for faster GPU load

3. ENVIRONMENT INIT
   └── TRANSFORMERS_OFFLINE=1
   └── CUDA_VISIBLE_DEVICES=0
   └── TRITON_PTXAS_PATH → custom ptxas binary

4. vLLM ENGINE INIT
   └── tensor_parallel_size=1
   └── max_lora_rank=32
   └── enable_prefix_caching=True
   └── enable_chunked_prefill=True
   └── gpu_memory_utilization=0.85

5. PROMPT CONSTRUCTION (per row)
   └── detect_category(prompt)
   └── Inject category-specific extra instructions
   └── apply_chat_template(enable_thinking=True)
   └── Wrap: "Solve step-by-step … Put final answer inside \boxed{}"

6. MULTI-SAMPLE GENERATION
   └── 5 independent runs with temperatures [0.6, 0.7, 0.8, 0.9, 1.0]
   └── top_p=0.9, max_tokens=7680

7. ANSWER EXTRACTION (per run)
   └── Priority 1: \boxed{...} — last non-empty match
   └── Priority 2: "Final answer is: ..."
   └── Priority 3: Short last line (≤20 chars, valid symbols)
   └── Priority 4: Last numeric match
   └── Fallback: "NOT_FOUND"

8. MAJORITY VOTING
   └── Force numeric for: gravity, numeral, unit_conversion
   └── Majority + shortest for all other categories
   └── Fallback to "0" if no valid answers

9. EVALUATION (optional)
   └── verify(): float comparison (rel_tol=1e-2) or string match
   └── Per-category stats: correct / total / weightage / contribution
   └── Exports: results.csv, validation.csv, mistakes/{category}.csv

10. SUBMISSION
    └── Pack adapter files → submission.zip
```

---

## ✍️ Prompt Engineering Strategy

Each puzzle type gets tailored system instructions injected into the user prompt before inference. This is a critical part of the performance strategy.

### General Template

```
Solve step-by-step carefully.

{original_prompt}
{category_specific_extra}

Put final answer inside \boxed{}
```

### Category-Specific Extras

**`bit_manipulation`** — Prevents common LLM failure of padding with leading zeros:
```
IMPORTANT for binary output:
- Output ONLY the resulting binary number
- Do NOT pad with leading zeros
- If the result starts with 0s, drop them
- Example: if result is 1000011, write 1000011 NOT 01000011
```

**`equation_numeric_deduce` / `equation_numeric_guess`** — Forces symbol-preserving answers:
```
STRICT RULES:
- Do NOT assume standard math
- Find transformation rule from ALL examples
- The answer may include operator symbols (like /, -, +, {, etc.)
- Do NOT strip operator symbols from your answer
- Verify your rule against every example before answering
- NEVER guess
```

**`cryptarithm_deduce` / `cryptarithm_guess`** — Encourages multi-pass reasoning:
```
Try multiple patterns before final answer. Do not guess immediately.
```

---

## 🗳️ Multi-Sample Voting

Instead of a single inference pass, the pipeline generates **5 independent answers** using dynamic temperature sampling, then aggregates:

```
Run 1 → temp=0.6 → answer_1
Run 2 → temp=0.7 → answer_2
Run 3 → temp=0.8 → answer_3
Run 4 → temp=0.9 → answer_4
Run 5 → temp=1.0 → answer_5
         │
         ▼
     Filter valid answers (not NOT_FOUND, len < 20)
         │
         ├── If category ∈ {gravity, numeral, unit_conversion}
         │   AND ≥3 numeric answers:
         │   → Counter(numeric).most_common(1)[0][0]
         │
         └── Else:
             → Sort by (-frequency, len) → pick shortest most-common
                 │
                 └── Fallback: "0"
```

The design favors shorter answers when frequencies tie — because LLMs tend to pad correct answers with units or symbols that break string matching.

---

## 🔬 Answer Extraction

The `extract_final_answer()` function implements a priority-ordered extraction cascade:

```python
Priority 1: \boxed{...}          ← Explicit LaTeX boxing (highest confidence)
Priority 2: "Final answer is: X" ← Structured phrasing
Priority 3: Short clean last line ← ≤20 chars, only valid symbols
Priority 4: Last numeric match   ← regex -?\d+(?:\.\d+)?
Fallback:   "NOT_FOUND"          ← Triggers fallback voting logic
```

**Verification** uses numeric tolerance for float answers:

```python
math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
```

String answers fall back to case-insensitive exact match.

---

## ⚙️ Configuration

| Parameter | Value | Notes |
|---|---|---|
| `MODEL_PATH` | `metric/nemotron-3-nano-30b-a3b-bf16` | Downloaded via `kagglehub` |
| `max_model_len` | `8192` | Max context window tokens |
| `max_tokens` | `7680` | Max generation tokens per call |
| `max_lora_rank` | `32` | LoRA adapter rank |
| `max_num_seqs` | `64` | Max concurrent sequences in vLLM |
| `gpu_memory_utilization` | `0.85` | Fraction of GPU VRAM for vLLM |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `temperatures` | `[0.6, 0.7, 0.8, 0.9, 1.0]` | One per sample run |
| `NUM_SAMPLES` | `5` | Independent generation passes |
| `EVALUATION_SAMPLE_SIZE` | `950` | Rows used for local eval |
| `cache_workers` | `16` | Threads for model cache warmup |
| `cache_chunk_mb` | `1024` | Chunk size for pre-read (1 GB) |

---

## 📊 Performance

Evaluation is run against the training set (first 950 rows) with per-category breakdowns:

| Category | Correct | Total | Weight | Accuracy | Contribution |
|---|---|---|---|---|---|
| bit_manipulation | — | — | ~17% | — | — |
| cipher | — | — | ~16% | — | — |
| equation_numeric | — | — | ~17% | — | — |
| cryptarithm | — | — | ~17% | — | — |
| gravity | — | — | ~11% | — | — |
| numeral | — | — | ~11% | — | — |
| unit_conversion | — | — | ~12% | — | — |
| **TOTAL** | **—** | **950** | **100%** | **—** | **—** |

> Results are filled at runtime into `results.csv`. Per-category mistake files are saved to `mistakes/{category}.csv`.

---

## 📁 Project Structure

```
.
├── nvidia-nemotron-model.ipynb   # Main notebook: inference + evaluation
├── adapter_config.json           # LoRA adapter configuration (extracted from zip)
├── submission.zip                # Packed LoRA adapter weights (output artifact)
├── results.csv                   # Per-category accuracy table (generated)
├── validation.csv                # Full row-level predictions (generated)
└── mistakes/
    ├── bit_manipulation.csv      # Wrong predictions per category
    ├── cipher.csv
    ├── equation_numeric_deduce.csv
    ├── equation_numeric_guess.csv
    ├── cryptarithm_deduce.csv
    ├── cryptarithm_guess.csv
    ├── gravity.csv
    ├── numeral.csv
    └── unit_conversion.csv
```

---

## 🚀 Getting Started

### Prerequisites

This notebook is designed for the **Kaggle competition environment** with GPU access. It assumes:

- A T4 or A100 GPU (single card, `CUDA_VISIBLE_DEVICES=0`)
- Access to the competition dataset at `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/`
- The metric utility scripts at `/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script/`

### Running the Notebook

1. **Clone or fork** this notebook into a Kaggle environment
2. **Attach** the competition dataset and the pretrained model dataset
3. **Place** your `submission.zip` (LoRA adapter) at the path set in `SUBMISSION_ZIP_PATH`
4. **Set** `RUN_EVALUATION = True` for local scoring, or `False` for submission-only mode
5. **Run all cells** — the pipeline handles everything from cache warmup to CSV export

### Evaluation Mode

```python
RUN_EVALUATION = True        # Score against training labels
EVALUATION_SAMPLE_SIZE = 950 # How many rows to evaluate
```

Set `RUN_EVALUATION = False` to skip local scoring and only generate `submission.zip`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `vllm` | High-throughput LLM inference engine |
| `kagglehub` | Model and dataset download from Kaggle |
| `pandas` | Data loading and results aggregation |
| `tqdm` | Progress bars |
| `torch` | PyTorch backend (uninstalled and replaced by vLLM's build) |
| `triton` | Custom GPU kernel backend (ptxas path overridden) |
| `zipfile` | Packing/unpacking LoRA adapter weights |

> ⚠️ The notebook explicitly **uninstalls** the default `torch` / `torchvision` / `torchaudio` and replaces them with vLLM-compatible builds via `uv pip`.

---

## 🔖 Labels

![NLP](https://img.shields.io/badge/NLP-Reasoning-blue?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Fine--tuned-purple?style=flat-square)
![LoRA](https://img.shields.io/badge/Adapter-LoRA-orange?style=flat-square)
![vLLM](https://img.shields.io/badge/Inference-vLLM-red?style=flat-square)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![NVIDIA](https://img.shields.io/badge/Hardware-NVIDIA%20GPU-76B900?style=flat-square&logo=nvidia)
![Nemotron](https://img.shields.io/badge/Model-Nemotron--3--Nano--30B-green?style=flat-square)
![BF16](https://img.shields.io/badge/Precision-BF16-yellow?style=flat-square)
![Majority Vote](https://img.shields.io/badge/Strategy-Majority%20Voting-lightblue?style=flat-square)
![Chain of Thought](https://img.shields.io/badge/Prompting-Chain--of--Thought-pink?style=flat-square)
![Symbolic Reasoning](https://img.shields.io/badge/Task-Symbolic%20Reasoning-darkgreen?style=flat-square)
![Cryptarithmetic](https://img.shields.io/badge/Puzzle-Cryptarithmetic-brown?style=flat-square)
![Bit Manipulation](https://img.shields.io/badge/Puzzle-Bit%20Manipulation-black?style=flat-square)
![Unit Conversion](https://img.shields.io/badge/Puzzle-Unit%20Conversion-teal?style=flat-square)
![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)
![Competition](https://img.shields.io/badge/Type-Competition%20Submission-gold?style=flat-square)
![Evaluation](https://img.shields.io/badge/Evaluation-Per--Category%20Accuracy-informational?style=flat-square)

---

<div align="center">

Made for the **NVIDIA Nemotron-3 Reasoning Challenge** on Kaggle

</div>
