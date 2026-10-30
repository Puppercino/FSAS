# Domain-Adaptive Sentiment & Emotion Analysis
### COS60011 – Technology Design Project · Swinburne University of Technology

A team project investigating how well emotion classifiers trained on general short-form text **transfer** to a different domain — first-person emotional narratives (ISEAR). This repository contains the live **Streamlit demo** and the full **experiment notebook**.

**Live demo:** https://qctpbztqdudwx3ka6qqsyt.streamlit.app/ 

---

## What the demo does

Type a short, first-person description of a situation and the model predicts the dominant emotion — **joy, sadness, anger, or fear** — with a probability chart for each class.

> Example: *"I was walking home alone at night when I heard footsteps behind me."* → **fear**

The deployed demo offers a selection of models trained using BERT, specifically: 
- `fsas-bert-domain`: Trained only on the target dataset.
- `fsas-bert-mixed`: Trained on both source and target datasets in one run.
- `fsas-bert-mixed-weighted`: Like BERT (Mixed), this model is trained on source and target in one run, but uses dataset weighting (oversampled by x3) for transfer learning.
- `fsas-bert-sequential`: Trained on a source dataset, then fine-tuned on a target dataset using transfer learning.

---

## The research question

Does knowledge learned from a **general (source)** domain help in a **specific (target)** domain, or does the domain shift hurt?

| | Dataset | Style |
|---|---|---|
| **Target (domain)** | [ISEAR](https://www.kaggle.com/datasets/dalopeza/isear-dataset) | Longer first-person narratives |
| **Source (general)** | [Additional dataset for emotions](https://www.kaggle.com/datasets/dimonyara/additional-dataset-for-emotions) | Short social-style text |

**Shared 4-class label space:** `joy`, `sadness`, `anger`, `fear`.
`fun → joy` and `worry → fear` are deliberate semantic merges (closest proxies); `love` and `surprise` are excluded.

---

## Approaches compared

- **Classical ML (TF-IDF):** Logistic Regression, SVM, Naive Bayes
  - **A** general data only · **B** domain data only · **C** mixed
- **BERT fine-tuning** (`bert-base-uncased`, transfer learning)
  - **S1** domain only · **S2** sequential (source → target) · **S3** mixed · **S4** weighted mixed (target ×3)
  - A separate source-only run is reported as BERT-A
- **LLM prompting (no training):** `flan-t5-large` — zero-shot, few-shot, structured

All models are scored on one frozen target test set that is never used for training, tuning or checkpoint selection. Duplicates are removed before splitting.

---

## Key results (macro F1 on the ISEAR test set)

| Model | Macro F1 | Accuracy |
|---|---|---|
| **BERT – S2 sequential (best)** | **0.828** | 0.829 |
| BERT – S1 domain only | 0.825 | 0.825 |
| BERT – S3 mixed | 0.820 | 0.821 |
| BERT – S4 weighted | 0.799 | 0.799 |
| LLM – few-shot | 0.786 | 0.788 |
| LLM – zero-shot | 0.787 | 0.787 |
| SVM – B domain only | 0.715 | 0.716 |
| Logistic Regression – B domain only | 0.708 | 0.708 |
| SVM – C mixed | 0.683 | 0.686 |
| BERT – A general only | 0.552 | 0.551 |
| SVM – A general only | 0.411 | 0.435 |

**Takeaways**
- Models trained only on the general data score high in-domain but drop sharply on ISEAR — a clear **domain gap** (e.g. SVM-A: 0.918 source F1 vs 0.411 target F1).
- Target-domain data matters most. Fine-tuning BERT first on the source and then on the target (S2) gave the best result, slightly ahead of target-only (S1).
- Simply mixing source and target data did not beat domain-only training.
- Transformers clearly outperform TF-IDF baselines (0.83 vs 0.72 F1).

Full tables, confusion matrices, error analysis and per-class recall are in the notebook.

---

## Repository structure

```
├── app.py                       # Streamlit demo
├── train_model.py               # trains the demo model (TF-IDF + LogReg on ISEAR)
├── requirements.txt             # dependencies for the deployed app
├── model.pkl                    # trained classifier (generated)
├── vectorizer.pkl               # fitted TF-IDF vectorizer (generated)
└── project_notebook_v2.ipynb    # full experiments: classical, BERT, LLM
```

---

## Run it locally

```bash
git clone https://github.com/Musfiq00/emotion-classifier.git
cd emotion-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

### Retrain the demo model (optional)

```bash
pip install kagglehub pandas scikit-learn joblib
python train_model.py
```

This downloads ISEAR from Kaggle and regenerates `model.pkl` and `vectorizer.pkl`. Hold-out performance for the demo model: **accuracy 0.73, macro F1 0.73**.

### Run the full experiments

Open `project_notebook_v2.ipynb` in Google Colab, switch to a **T4 GPU** runtime before Section 5 (BERT), and run the cells top to bottom.

---

## Limitations

- The demo handles **English** text in four emotion classes only.
- It works best on first-person, ISEAR-style narratives, and can fail on **sarcasm** and very short or out-of-style text.
- `fun → joy` and `worry → fear` are approximations, so cross-dataset comparisons carry label-mapping noise.

---

## Team

| Name | Student ID |
|---|---|
| `Jordan Lyall ` | `106594525` |
| `Masum Musfiq` | `106230751` |
| `Davit Ou` | `106224754` |
| `Sokun Sor` | `106213341` |

Built with Python, scikit-learn, Hugging Face Transformers, and Streamlit.
