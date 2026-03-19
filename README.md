# iKPUB

Next generation automatic publication classifier for WMKO

## Setup

install modules in development mode using

```zsh
pip install -e .
```

### Section Specific Setup
- [Query ADS Setup](#query-ads-setup)
- [Publication Prep Setup](#publication-prep-setup)

## 1. Query ADS

### Query ADS Setup

1. Create .env at project root with ADS_TOKEN
2. Create a SQLite database at data/pubs/kpub.db

### Run Query

Query Publications for KPUB (Manually specify year range)
```zsh
python src/data_pipes/query_ads.py --query kpub --start-year 2000 --end-year 2025
```

Query Publications for KOA
```zsh
python src/data_pipes/query_ads.py --query koa --start-year 2008 --end-year 2025
```

## 2. Fetch Full Text

Fetch full text for KPUB publications
```zsh
python src/data_pipes/fetch_full_text.py --start-year 2000 --end-year 2025
```

Fetch full text for KOA publications
```zsh
python src/data_pipes/fetch_full_text.py --table koa --start-year 2008 --end-year 2025
```

## 3. Prepare Publications

Add Manual Labels and full text to the kpub SQLite database table

### Publication Prep Setup

Get a copy of manual_kpub.db from code maintainers and place it in `data/pubs/manual_kpub.db`

### Run code

Prepare the publications table for KPUB classification with manual labels and full text
```zsh
python src/data_pipes/prepare.py
```

Prepare the koa table for KOA classifications with manual labels and full text
```zsh
python src/data_pipes/prepare.py --table koa
```

## 4. Train/Test Model 

