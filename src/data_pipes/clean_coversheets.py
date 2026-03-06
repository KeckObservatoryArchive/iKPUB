"""Parse coversheet .dat files and output formatted JSON files.

Usage:
    python src/data_pipes/clean_coversheets.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "coversheets"
PREFIX = "KPUB_DATA: "


def parse_dat_file(dat_path: str) -> list[dict]:
    """Read a .dat file and extract the JSON data after the KPUB_DATA prefix."""
    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if not content.startswith(PREFIX):
        raise ValueError(f"Expected file to start with '{PREFIX}': {dat_path}")

    json_str = content[len(PREFIX):]
    return json.loads(json_str)


def dat_to_json(dat_path: str) -> str:
    """Convert a .dat file to a .json file. Returns the output path."""
    data = parse_dat_file(dat_path)

    dat_path = Path(dat_path)
    # e.g. data_2018_410_noENG.dat -> coversheets_2018.json
    year = dat_path.name.split("_")[1]
    out_path = dat_path.parent / f"coversheets_{year}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return str(out_path)


def main():
    dat_files = sorted(DATA_DIR.glob("*.dat"))
    if not dat_files:
        print("No .dat files found in", DATA_DIR)
        return

    for dat_path in dat_files:
        out_path = dat_to_json(dat_path)
        print(f"{dat_path.name} -> {Path(out_path).name}")


if __name__ == "__main__":
    main()
