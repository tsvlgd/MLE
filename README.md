# MLE — Machine Learning Engineering from First Principles

**A structured, scalable personal repository for advanced ML learning, research, and hands-on implementations.**

This repository evolves continuously as I deepen my understanding of machine learning fundamentals. It's organized into a **canonical learning track** (`ml/`) with polished tutorials and implementations, and an **active workbench** (`advanced_ml/`) for ongoing exploration and advanced coursework.

## 🎯 Philosophy

- **From first principles:** Understand *why* before implementing *how*.
- **Canonical reference:** Separate stable, reader-friendly content from experimental work.

---

## 📁 Repository Structure

The repo now follows a **numbered-topic layout** inspired by advanced learning repositories. Redundancy has been removed and structure clarified:

```
MLE/
├── README.md                          # This file
├── LICENSE
├── datasets/                          # Datasets (small CSVs + pointers to external sources)
├── tools/                             # Helpers: scaffold_topics.py, fix_notebook_links.py
│
├── advanced_ml/                       # Active workbench (in-progress deep dives)
└── ml/                                # Canonical learning material 
```

---

## 🗺️ How to Navigate

### For Learning & Stable Content
Start here if you want polished tutorials or reliable implementations:
- **Tutorials:** `ml/01-docs/docs/`
- **Algorithm implementations:** `ml/02-implementations/implementations/`
- **Projects:** `ml/05-projects/projects/`

### For Experimentation & Advanced Topics
Explore here for in-progress work and active learning:
- **Neural networks & deep learning:** `advanced_ml/01-neural-nets/`
- **PyTorch & OpenCV:** `advanced_ml/02-pytorch-opencv/`
- **Archive:** `advanced_ml/03-archive/` (completed or shelved topics)

### For Data & Tools
- **Datasets:** `datasets/` (small CSVs stored; large data linked externally)
- **Repo utilities:** `tools/` (scaffolding, link-fixing scripts)

---

## 📊 Key Materials

| Topic                               | Location                                                          | Status      |
| ----------------------------------- | ----------------------------------------------------------------- | ----------- |
| Neural Networks (from scratch)      | `advanced_ml/01-neural-nets/`                                     | In progress |
| Dimensionality Reduction (PCA, SVD) | `ml/02-implementations/implementations/dimensionality_reduction/` | Stable      |
| Ensemble Learning                   | `ml/02-implementations/implementations/ensembles/`                | Stable      |
| Time Series & Pipelines             | `ml/05-projects/projects/time_series/`                            | Stable      |
| Kaggle Exercises                    | `ml/04-kaggle/kaggle/intermediateML/`                             | Stable      |

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Savvythelegend/MLE.git
   cd MLE
   ```

2. **Explore a specific topic:**
   ```bash
   # For tutorials
   ls ml/01-docs/docs/
   
   # For implementations
   ls ml/02-implementations/implementations/
   ```

3. **Start a Jupyter notebook:**
   ```bash
   jupyter notebook ml/02-implementations/implementations/regression/
   ```

---

## 🔧 Tools Included

- **`tools/scaffold_topics.py`** — Create numbered topic folders and README templates (dry-run by default).
- **`tools/fix_notebook_links.py`** — Update plain-text references in `.md` and `.ipynb` files after moves (dry-run; use `--apply` to write).

---

## 📝 Contributing

- **Keep experimental work in `advanced_ml/`** while moving polished tutorials to `ml/`.
- **Use `git mv`** to preserve file history.
- **Add a `README.md`** in any new topic folder explaining its purpose and status.
- **Run the link-fixer** after moving files to prevent broken references.

---

## 📜 License

This repository is available under the terms of the `LICENSE` file.

---

**Last updated:** November 2025  
**Structure:** Numbered topics with canonical separation of stable content and active workbenches.
