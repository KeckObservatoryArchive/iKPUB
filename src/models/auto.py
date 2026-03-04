"""

"""
from models.base_kpub_classifier import KPUBClassifier
import pandas as pd

class AutoClassifier(KPUBClassifier):
    
    def train(self, X_train, y_train):
        pass
    
    def predict(self, X_test):
        return pd.Series([0] * len(X_test))
