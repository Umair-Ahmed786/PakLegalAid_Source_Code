# PakLegalAid — FastAPI Backend & ML Experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)

> Source code accompanying the paper submitted to **PLOS ONE**.

**PakLegalAid** is an AI-powered legal question-answering system focused on Pakistani law. This repository contains two components: the **FastAPI backend** that serves the trained models as a REST API, and the **ML experiment notebooks** used to train and evaluate those models.

The ML pipeline combines intent classification (legal vs. non-legal query filtering), supervised fine-tuning of LLaMA 3.2 3B Instruct with LoRA, and a FAISS-based Retrieval-Augmented Generation (RAG) system built from seven Pakistani legal statutes.

---

## Repository Structure

```
PakLegalAid_FastApi_backend/
├── Fastapi/                                          # FastAPI backend — serves the trained models as REST API endpoints
├── classification_experiments_on_various_models.py  # Binary classification: legal vs. non-legal queries (DistilBERT, RoBERTa)
├── llama_3_2_3b_with_without_ft.py                  # LLaMA 3.2 3B baseline vs. fine-tuned comparison + FAISS vector DB builder
├── finetuning_llama_3_2_instruct.py                 # Core SFT pipeline using Unsloth + LoRA (rank=16)
├── copy_of_llama_3_2_3b_with_lora.py                # Variant LoRA fine-tuning run for full reproducibility
├── rag_finetuned_model.py                            # RAG pipeline: FAISS index over Pakistani legal statutes + augmented generation
└── README.md
```

---

## Datasets

Datasets are **not included** in this repository due to size and licensing constraints.

| Dataset | Used In |
|---|---|
| `legal_100_Queries_cleaned.csv` — 100 lawyer-curated legal Q&A pairs | Experiments 2, 3, 4 |
| `Classification_dataset.csv` — binary-labeled legal/non-legal questions | Experiment 1 |
| Pakistani Legal Statutes (CSVs) — Limitation Act 1908, CrPC, Muslim Family Laws Ordinance 1961, Pakistan Penal Code, Police Law, Transfer of Property Act, Qanun-e-Shahadat 1984 | Experiment 5 |

---

## Key ML Configurations

**LoRA Fine-Tuning:**
```python
r = 16, lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
use_gradient_checkpointing = "unsloth"
```

**Training:**
```python
per_device_train_batch_size=2, gradient_accumulation_steps=4,
max_steps=60, learning_rate=2e-4, optim="adamw_8bit"
```


---

## Evaluation

- **Generative models:** BLEU, ROUGE-1/2/L, METEOR
- **Classification models:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Running the ML Experiments

All experiment notebooks are designed for **Google Colab** with a T4/A100 GPU. Install dependencies with:

```bash
pip install unsloth transformers datasets evaluate rouge_score nltk \
            langchain_community faiss-cpu sentence-transformers trl seaborn
```

Upload the required dataset CSV files to `/content/` before running. Each notebook is self-contained and runs end-to-end sequentially.

## Running the FastAPI Backend

```bash
cd Fastapi
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs will be available at `http://localhost:8000/docs`.

---

## Citation

```bibtex
@article{yourpaper2025,
  title   = {Paper Title Here},
  author  = {Author Names},
  journal = {PLOS ONE},
  year    = {2025},
  doi     = {10.xxxx/journal.pone.xxxxxxx}
}
```

---

## Acknowledgements

[Unsloth](https://github.com/unslothai/unsloth) · [Hugging Face](https://huggingface.co/) · [LangChain](https://www.langchain.com/) · [FAISS](https://faiss.ai/) · Lawyer experts who curated the Q&A dataset
