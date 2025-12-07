### **MLE — Machine Learning Engineering from First Principles**
**Structure:**
```
MLE/
├── README.md                # Overview and setup
├── requirements.txt         # Dependencies
│
├── advanced_ml/             # Experimental work (in-progress)
│   ├── 01-neural-nets/      # Neural networks from scratch
│   └── 02-pytorch-opencv/   # PyTorch and OpenCV deep dives
│
├── ml/                      # Stable, polished content
│   ├── 01-docs/             # Tutorials and conceptual notes
│   ├── 02-implementations/  # Algorithm implementations (e.g., regression, PCA)
│   ├── 04-kaggle/           # Kaggle exercises (e.g., Intermediate ML)
│   └── 05-projects/         # End-to-end projects (e.g., time series, NLP)
│
├── datasets/                # Datasets (small files + external links)
│   └── archive/             # Example: heart_failure_clinical_records_dataset.csv
│
├── programming/             # Language-specific resources
│   ├── python/              # Python for ML (OOP, modularization)
│   └── sql/                 # SQL scripts for data science
│
└── tools/                   # Utility scripts
    ├── scaffold_topics.py   # Create numbered topic folders
    └── fix_notebook_links.py # Update references in `.md`/`.ipynb` files
```

---

### **Key Areas**
1. **Stable Content (`ml/`)**
   - Tutorials: `ml/01-docs/`
   - Implementations: `ml/02-implementations/`
   - Projects: `ml/05-projects/`

2. **Experimental Work (`advanced_ml/`)**
   - Neural networks: `advanced_ml/01-neural-nets/`
   - PyTorch/OpenCV: `advanced_ml/02-pytorch-opencv/`

3. **Utilities (`tools/`)**
   - Scaffolding: `scaffold_topics.py`
   - Link fixer: `fix_notebook_links.py`

---

### **Quick Start**
```bash
git clone https://github.com/Savvythelegend/MLE.git
cd MLE
jupyter notebook ml/02-implementations/implementations/regression/logistic_regression.ipynb
```
