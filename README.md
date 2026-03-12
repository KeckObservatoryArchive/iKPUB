# iKPUB

Next generation AI-assist KPUB for WMKO

## Setup

install modules in development mode using

```zsh
pip install -e .
```

## Setup for query_ads.py

1. Create .env at root with ADS_TOKEN
2. Create SQLite database at data/pubs/kpub.db

Run query_ads.py to populate table `publications`
