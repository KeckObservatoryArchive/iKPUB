"""
Rule-based auto classifier for Keck publications.
Ports the classification logic from kpub.py into a sklearn-compatible interface.

Classification happens in two layers (mirroring kpub.py):
  1. _apply_exclusion_rules  -> like update()      : hard excludes (no abstract, proposal bibcodes)
  2. _find_snippets           -> like add_article() : searches full text fields for keywords;
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


class AutoClassifier(KPUBClassifier):

    def __init__(self, instruments=None, acknowledgements=None, archive=None):
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
    # Layer 1 — Hard exclusions (mirrors update())
    # ------------------------------------------------------------------

    def _apply_exclusion_rules(self, row):
        """
        Returns True if the article should be hard-excluded before any
        further classification (no abstract, proposal/cospar/tmp bibcodes).

        Mirrors the exclusion block inside kpub.update().
        """
        abstract = row.get('abstract')
        if not abstract or str(abstract).strip() == '':
            return True

        bibcode = str(row.get('bibcode', ''))
        for pattern in BIBCODE_EXCLUSION_PATTERNS:
            if pattern in bibcode:
                return True

        return False


    # ------------------------------------------------------------------
    # Layer 2 — Snippet search (mirrors add_article())
    # ------------------------------------------------------------------

    def _find_snippets(self, row):
        """
        Searches available text fields for instrument names, acknowledgement
        strings, and archive strings.

        NOTE: In kpub.py this searches the full PDF text. Here we use the ADS
        metadata fields available in X_test as a proxy: ack, abstract,
        facility, aff.

        Returns
        -------
        dict : {matched_word: {'count': int, 'snippets': [str]}}
               Empty dict means nothing relevant was found → unrelated.
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
        """
        Searches for a word/phrase in text, returning context snippets.
        Mirrors the regex approach in kpub's get_word_match_counts_by_pdf().
        """
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