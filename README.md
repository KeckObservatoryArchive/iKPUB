# iKPUB

Automatic publication classifier for W. M. Keck Observatory.

## Setup

Install in development mode:
```zsh
pip install -e .
```

Create `.env` at the project root:
```
# Not needed if you are working from the MongoDB option
ADS_TOKEN=your_token_here

MONGO_SERVER=hostname
MONGO_PORT=27017
MONGO_USER=username
MONGO_PWD=password
```

---

## Option A: MongoDB Pathway (Recommended)

Articles live in MongoDB. Full text stays on the filesystem (`data/pubs/full_text/`). Predictions write directly back to MongoDB.

### 1. Fetch Full Text

Reads bibcodes and `links_data` from MongoDB, downloads PDFs, extracts text to `data/pubs/full_text/{year}/{bibcode}.txt`.

```zsh
python -m src.data.fetch_full_text --mongo --collection test_articles --year 2024
python -m src.data.fetch_full_text --mongo --collection test_articles --start-year 2020 --end-year 2025
```

### 2. Train / Test

Loads articles from MongoDB, derives `keck_manual` from the `affiliation` field (`"keck"` → positive, everything else → negative), merges full text from the filesystem, and runs the standard train/test split.

```zsh
python src/eval/test_harness.py transformer --mongo --year 2000-2023 --save
python src/eval/test_harness.py embedding --mongo --collection test_articles
```

Docs without an `affiliation` set are skipped and reported in the run summary.

### 3. Predict Labels

Loads articles from MongoDB, merges full text from filesystem, runs classifiers, writes predictions back to MongoDB as flat fields (`ilabel`, `keck_score`, `idrp`, `drp_reason`, `ikoa`, `koa_reason`).

```zsh
# Keck classification (transformer)
python -m src.eval.predict_labels 2024 --mongo --collection test_articles --task keck

# DRP classification (LLM, runs on keck-positive papers only)
python -m src.eval.predict_labels 2024 --mongo --collection test_articles --task drp

# KOA classification (LLM)
python -m src.eval.predict_labels 2024 --mongo --collection test_articles --task koa
```

---

## Option B: SQLite Pathway

The original pipeline. All data lives in `data/pubs/kpub.db`.

### 1. Ingest Data

Query ADS, fetch full text, and prepare labeled data. Requires `data/pubs/manual_kpub.db` — get a copy from code maintainers.

```zsh
python src/data/ingest.py --query keck --start-year 2000 --end-year 2025
python src/data/ingest.py --query koa --start-year 2008 --end-year 2025
```

Use `--skip-query` or `--skip-fulltext` to re-run individual stages without repeating earlier ones.

### 2. Train / Test

```zsh
python src/eval/test_harness.py transformer --table keck --save
python src/eval/test_harness.py embedding --table koa
```

Available models: `transformer`, `embedding`, `snippet` (rule-based). Hyperparameters in `config/models.yaml`.

### 3. Predict Labels

```zsh
python -m eval.predict_labels 2024
python -m eval.predict_labels 2020-2024 --model-path data/models/trained/my_model
```

### 4. Export to MongoDB

```zsh
python -m data.sqlite_to_mongo 2024
```
