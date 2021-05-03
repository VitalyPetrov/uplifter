import numpy as np
import pandas as pd
from typing import Dict
from sklift.metrics import (
    uplift_auc_score,
    perfect_uplift_curve,
    uplift_curve,
    qini_curve,
    perfect_qini_curve,
    qini_auc_score
)


class BaseUpliftModel:
    def __init__(
        self,
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame,
        treatment_nm: str,
        target_nm: str

    ):
        self.train, self.test = training_dataset, testing_dataset

        self.treatment_nm = treatment_nm
        self.target_nm = target_nm

        if not set(self.train.columns) == set(self.test.columns):
            raise RuntimeError('Columns at train and test subsets are mismatched')

        self.feature_columns = [
            c for c in self.train.columns
            if c not in (self.treatment_nm, self.target_nm)
        ]

    def fit(self, features, **kwargs) -> None:
        pass

    def predict(self, features) -> np.ndarray:
        pass

    def compute_metrics(self) -> Dict[str, float]:
        args = {
            'y_true': self.test[self.target_nm],
            'uplift': self.predict(self.test[self.feature_columns]),
            'treatment': self.test[self.treatment_nm]
        }

        return {
            'uplift_auc_score': uplift_auc_score(**args),
            'qini_auc_score': qini_auc_score(**args)
        }
