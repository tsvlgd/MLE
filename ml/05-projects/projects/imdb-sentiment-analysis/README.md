# Sentiment Analysis (IMDb)

This repository contains code and notebooks for training a sentiment analysis model on the IMDb dataset.

Repository layout (production-ready):

- src/             - Python source code (modules/scripts)
- notebooks/       - Jupyter notebooks (analysis, experiments)
- data/            - datasets and generated CSVs
- requirements.txt - Python dependencies

Quick start
1. Create and activate a virtual environment (zsh):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data
- Put the raw IMDb dataset under `data/aclImdb` (train/test pos/neg) or download it and extract to that path.

3. Generate the CSV (run the loader)

```bash
python -m src.load_data
```

This will create `data/movie_data.csv` (shuffled) which the notebooks expect.

Notebooks
- `notebooks/data_cleaning.ipynb` — data preprocessing and model training pipeline. See `notebooks/README.md` for more details.

Notes
- Don't commit the `venv/` folder or large raw datasets. Add `data/` entries to `.gitignore` if you don't want to commit dataset files.
