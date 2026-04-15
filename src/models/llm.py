"""LLM-based zero-shot classifier for Keck publications.

Uses a local LLM via Ollama to classify publications based on a prompt.
No training required -- classification is done via prompt engineering and
token logprobs for calibrated probability scores.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import ollama

from .base_kpub_classifier import KPUBClassifier
from data.compose import COMPOSE_FN

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_RETRIES = 3

TASK_PROMPTS = {
    "drp": PROMPTS_DIR / "drp_classify.txt",
    "koa": PROMPTS_DIR / "koa_classify.txt",
    "keck": PROMPTS_DIR / "keck_classify.txt",
}

TASK_QUESTIONS = {
    "drp": "Did this publication use a Keck Observatory Data Reduction Pipeline?",
    "koa": "Did this publication use data from the Keck Observatory Archive (KOA)?",
    "keck": "Is this a Keck Observatory publication?",
}


class LLMClassifier(KPUBClassifier):

    def __init__(
        self,
        model_name: str = "gemma4:12b",
        host: str = "http://localhost:11434",
        extraction_mode: str = "sentence",
        table: str = "keck",
        task: str = "drp",
        temperature: float = 0.0,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 8192,
        threshold: float = 0.5,
        prompt_path: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.host = host
        self.extraction_mode = extraction_mode
        self.table = table
        self.task = task
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.threshold = threshold
        self.system_prompt = self._load_prompt(prompt_path)
        self.question = TASK_QUESTIONS.get(self.task, f"Does this publication match the '{self.task}' criteria?")

    def _load_prompt(self, prompt_path: str | None = None) -> str:
        if prompt_path:
            return Path(prompt_path).read_text().strip()
        path = TASK_PROMPTS.get(self.task)
        if path is None:
            raise ValueError(f"Unknown task '{self.task}'. Choose from: {', '.join(TASK_PROMPTS)}")
        return path.read_text().strip()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """No training needed for a zero-shot LLM classifier."""
        pass

    def predict(self, X_test: pd.DataFrame, return_proba: bool = False) -> pd.Series:
        compose = COMPOSE_FN[self.table]
        client = ollama.Client(host=self.host, timeout=120)
        results = []

        for _, row in tqdm(X_test.iterrows(), total=len(X_test), desc="LLM classify"):
            text = compose(row, extraction_mode=self.extraction_mode)
            prob, _ = self._classify_one(client, text)
            results.append(prob if return_proba else (1 if prob >= self.threshold else 0))

        return pd.Series(results, index=X_test.index)

    def predict_with_reasons(self, X_test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Classify and return (scores, reasons) for each publication."""
        compose = COMPOSE_FN[self.table]
        client = ollama.Client(host=self.host, timeout=120)
        scores = []
        reasons = []

        for _, row in tqdm(X_test.iterrows(), total=len(X_test), desc="LLM classify"):
            text = compose(row, extraction_mode=self.extraction_mode)
            prob, response_text = self._classify_one(client, text)
            scores.append(prob)
            reasons.append(response_text)

        return pd.Series(scores, index=X_test.index), pd.Series(reasons, index=X_test.index)

    def _classify_one(self, client, text: str) -> tuple[float, str]:
        """Classify a single publication. Returns (P(yes), response_text)."""
        for attempt in range(MAX_RETRIES):
            truncated = text[:int(len(text) * (1 - 0.1 * attempt))]
            if attempt % 2 == 0:
                user_content = f"Classify the following publication:\n\n{truncated}\n\n{self.question}"
            else:
                user_content = f"{self.question}\n\n{truncated}"

            try:
                response = client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    options={
                        "temperature": self.temperature,
                        "num_ctx": self.max_input_tokens,
                    },
                )
            except Exception as e:
                logger.warning("Ollama request failed (attempt %d): %s", attempt + 1, e)
                continue

            content = response.get("message", {}).get("content", "").strip()
            if content:
                return self._extract_probability(response), content

        return 0.05, ""

    def _extract_probability(self, response) -> float:
        """Extract P(yes) from the response by parsing the first Yes/No line."""
        content = response.get("message", {}).get("content", "").strip().lower()

        for line in content.split("\n"):
            line = line.strip().lower().rstrip(".,!:;")
            line = re.sub(r"^[\-\*#>\s]+", "", line)  # strip markdown prefixes
            if line.startswith("yes"):
                return 0.95
            if line.startswith("no"):
                return 0.05
        return 0.05  # ambiguous/empty → no

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "host": self.host,
            "extraction_mode": self.extraction_mode,
            "table": self.table,
            "task": self.task,
            "temperature": self.temperature,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "threshold": self.threshold,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LLMClassifier":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        return cls(**meta)
