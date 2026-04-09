# iKPUB

Automatic publication classifier for W. M. Keck Observatory.

## Setup

Install in development mode:
```zsh
pip install -e .
```

Create `.env` at the project root:
```
ADS_TOKEN=your_token_here

# Only needed for step 4 (MongoDB export)
MONGO_SERVER=hostname
MONGO_PORT=27017
MONGO_USER=username
MONGO_PWD=password
```

## 1. Ingest Data

Query ADS, fetch full text, and prepare labeled data in one step. Requires `data/pubs/manual_kpub.db` — get a copy from code maintainers.

```zsh
python src/data/ingest.py --query keck --start-year 2000 --end-year 2025
python src/data/ingest.py --query koa --start-year 2008 --end-year 2025
```

Use `--skip-query` or `--skip-fulltext` to re-run individual stages without repeating earlier ones. The sub-scripts (`query_ads.py`, `fetch_full_text.py`, `prepare.py`) can still be run individually.

## 2. Train / Test

Train a classifier and evaluate on a held-out test set.

```zsh
python src/eval/test_harness.py transformer --table keck --save
python src/eval/test_harness.py embedding --table koa
```

Available models: `transformer`, `embedding`, `snippet` (rule-based).

Model hyperparameters live in `config/models.yaml` (see `config/models.default.yaml` for defaults). To run a batch of configs:

```zsh
python src/eval/run_queue.py --save
```

Queue files go in `config/queue/`.

## 3. Predict Labels

Run a trained model on publications for a year or year range and write predictions to the `predictions` table in `kpub.db`.

```zsh
python -m eval.predict_labels 2024
python -m eval.predict_labels 2020-2024 --model-path data/models/trained/my_model
```

## 4. Export to MongoDB

Push predictions from SQLite to MongoDB.

```zsh
python -m data.sqlite_to_mongo
python -m data.sqlite_to_mongo 2024
```
