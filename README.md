# iKPUB

Automatic publication classifier for W. M. Keck Observatory.

## Setup

Install in development mode:
```zsh
pip install -e .
```

Create `.env` at the project root:
```
MONGO_SERVER=hostname
MONGO_PORT=27017
MONGO_USER=username
MONGO_PWD=password
```

---

Articles live in MongoDB. Full text stays on the filesystem (`data/pubs/full_text/`). Predictions write directly back to MongoDB.

### 1. Fetch Full Text

Reads bibcodes and `links_data` from MongoDB, downloads PDFs, extracts text to `data/pubs/full_text/{year}/{bibcode}.txt`.

```zsh
python src/data/fetch_full_text.py --collection test_articles --year 2024
python src/data/fetch_full_text.py --collection test_articles --start-year 2020 --end-year 2025
```

### 2. Train / Test

Loads articles from MongoDB, derives `keck_manual` from the `affiliation` field (`"keck"` → positive, everything else → negative), merges full text from the filesystem, and runs the standard train/test split.

```zsh
python src/scripts/train.py transformer --year 2000-2023 --save
python src/scripts/train.py embedding --collection test_articles
python src/scripts/train.py transformer --no-test --save  # train on all labeled data
```

Docs without an `affiliation` set are skipped and reported in the run summary.

Available models: `transformer`, `embedding`, `snippet` (rule-based), `llm`. Hyperparameters in `config/models.yaml`.

To seed a fresh collection with broad-query training examples, see `scratch/insert_training_data.py`.

### 3. Predict Labels

Loads articles from MongoDB, merges full text from filesystem, runs classifiers, writes predictions back to MongoDB as flat fields (`ilabel`, `keck_score`, `idrp`, `drp_reason`, `ikoa`, `koa_reason`).

```zsh
# Keck classification (transformer)
python -m src.scripts.predict 2024 --collection test_articles --task keck

# DRP classification (LLM, runs on keck-positive papers only)
python -m src.scripts.predict 2024 --collection test_articles --task drp

# KOA classification (LLM)
python -m src.scripts.predict 2024 --collection test_articles --task koa
```
