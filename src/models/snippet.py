"""Rule-based snippet classifier for Keck publications.

Classification happens in two layers (based on the original kpub.py project):
  1. _apply_exclusion_rules : hard excludes (no abstract, proposal bibcodes)
  2. _find_snippets         : searches text fields for keywords;
                              any match → keck, no match → unrelated
"""

from models.base_kpub_classifier import KPUBClassifier
import pandas as pd
import re


# Default keyword lists — sourced directly from config.live.yaml
DEFAULT_INSTRUMENTS = [
    'DEIMOS',
    'ESI',
    'HIRES',
    'KCWI',
    'LRIS',
    'MOSFIRE',
    'NIRC2',
    'NIRES',
    'NIRSPEC',
    'OSIRIS',
    'AO',
    'LGS',
    'NGS',
]

# Strings that indicate a Keck acknowledgement in the text
DEFAULT_ACKNOWLEDGEMENTS = [
    'keck observatory',
    'W. M. Keck Observatory',
    'WMKO',
]

# Archive strings — presence indicates data archiving was referenced
DEFAULT_ARCHIVE = [
    'KOA',
    'Keck Observatory Archive',
]

# Bibcode patterns that are never real publications
BIBCODE_EXCLUSION_PATTERNS = ['.prop.', 'cosp..', '.tmp']


class SnippetClassifier(KPUBClassifier):

    def __init__(self, instruments=None, acknowledgements=None, archive=None, **kwargs):
        """
        Parameters
        ----------
        instruments : list of str, optional
            Instrument name strings to search for in text fields.
            Defaults to DEFAULT_INSTRUMENTS.
        acknowledgements : list of str, optional
            Acknowledgement strings indicating Keck usage (e.g. 'W. M. Keck Observatory').
            Defaults to DEFAULT_ACKNOWLEDGEMENTS.
        archive : list of str, optional
            Strings indicating data archive usage (e.g. 'KOA').
            Defaults to DEFAULT_ARCHIVE.
        """
        self.instruments      = instruments      or DEFAULT_INSTRUMENTS
        self.acknowledgements = acknowledgements or DEFAULT_ACKNOWLEDGEMENTS
        self.archive          = archive          or DEFAULT_ARCHIVE


    def train(self, X_train, y_train):
        """No training needed for a rule-based classifier."""
        pass


    def predict(self, X_test):
        """
        Predicts whether each article is a Keck publication (1) or not (0).

        Parameters
        ----------
        X_test : pd.DataFrame
            Must contain at minimum: bibcode, abstract, ack, facility, aff.

        Returns
        -------
        pd.Series of int (1 = keck, 0 = unrelated)
        """
        results = []
        for _, row in X_test.iterrows():
            label = self._classify_article(row)
            results.append(label)
        return pd.Series(results, index=X_test.index)


    # ------------------------------------------------------------------
    # Layer 1 — Hard exclusions
    # ------------------------------------------------------------------

    def _apply_exclusion_rules(self, row):
        """Return True if the article should be excluded (no abstract, proposal bibcodes)."""
        abstract = row.get('abstract')
        if not abstract or str(abstract).strip() == '':
            return True

        bibcode = str(row.get('bibcode', ''))
        for pattern in BIBCODE_EXCLUSION_PATTERNS:
            if pattern in bibcode:
                return True

        return False


    # ------------------------------------------------------------------
    # Layer 2 — Snippet search
    # ------------------------------------------------------------------

    def _find_snippets(self, row):
        """Search text fields for instrument, acknowledgement, and archive strings.

        Returns dict of {matched_word: {'count': int, 'snippets': [str]}}.
        Empty dict → unrelated.
        """
        text_fields = ['ack', 'abstract', 'facility', 'aff']
        combined_text = ' '.join(
            str(row.get(field, '') or '') for field in text_fields
        )

        words_to_search = self.instruments + self.acknowledgements + self.archive
        counts = {}
        for word in words_to_search:
            matches = self._find_word_in_text(word, combined_text)
            if matches:
                counts[word] = {
                    'count': len(matches),
                    'snippets': matches
                }
        return counts


    def _find_word_in_text(self, word, text):
        """Search for a word/phrase in text, returning context snippets."""
        snippets = []
        pattern = re.compile(
            r'(?:^|[\s/(\-:])' + re.escape(word),
            re.IGNORECASE
        )
        for match in pattern.finditer(text):
            start = max(0, match.start() - 80)
            end   = min(len(text), match.end() + 80)
            snippets.append(text[start:end])
        return snippets


    # ------------------------------------------------------------------
    # Main classification pipeline
    # ------------------------------------------------------------------

    def _classify_article(self, row):
        """
        Runs both classification layers for a single article row.

        Returns
        -------
        int : 1 if keck, 0 if unrelated
        """
        if self._apply_exclusion_rules(row):
            return 0

        snippets = self._find_snippets(row)
        return 1 if len(snippets) > 0 else 0