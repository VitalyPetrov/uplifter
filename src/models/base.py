import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Union
from sklift.metrics import (
    uplift_auc_score,
    perfect_uplift_curve,
    uplift_curve,
    qini_curve,
    perfect_qini_curve,
    qini_auc_score,
)

from src.conf import settings


class BaseUpliftModel:
    def __init__(
        self,
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame,
        treatment_nm: str = settings.data.treatment_nm,
        target_nm: str = settings.data.target_nm,
    ):
        self.train, self.test = training_dataset, testing_dataset

        self.treatment_nm = treatment_nm
        self.target_nm = target_nm

        if not set(self.train.columns) == set(self.test.columns):
            raise RuntimeError(
                "Columns at train and test subsets are mismatched"
            )

        self.feature_columns = [
            c
            for c in self.train.columns
            if c not in (self.treatment_nm, self.target_nm)
        ]

    def fit(self, features, **kwargs) -> None:
        pass

    def predict(self, features) -> np.ndarray:
        pass

    def compute_metrics(self) -> Dict[str, float]:
        args = {
            "y_true": self.test[self.target_nm],
            "uplift": self.predict(self.test[self.feature_columns]),
            "treatment": self.test[self.treatment_nm],
        }

        return {
            "uplift_auc_score": uplift_auc_score(**args),
            "qini_auc_score": qini_auc_score(**args),
        }

    @staticmethod
    def draw_uplift_curve(
        target_true: Union[pd.Series, np.ndarray],
        uplift_predicted: np.ndarray,
        treatment_true: Union[pd.Series, np.ndarray],
        with_perfect: bool = True
    ) -> None:
        plt.title('Uplift curve')
        if with_perfect:
            curve_perfect = perfect_uplift_curve(
                y_true=target_true, treatment=treatment_true
            )
            plt.plot(
                *curve_perfect, **settings.model.plotting_cfg, label='perfect'
            )

        curve_model = uplift_curve(
            y_true=target_true,
            uplift=uplift_predicted,
            treatment=treatment_true
        )

        plt.plot(
            *curve_model,
            **settings.model.plotting_cfg,
            label='model'
        )

        plt.plot(
            (curve_model[0][0], curve_model[0][-1]),
            (curve_model[1][0], curve_model[1][-1]),
            **settings.model.plotting_cfg,
            label='random'
        )

        plt.show()

    @staticmethod
    def draw_qini_curve(
        target_true: Union[pd.Series, np.ndarray],
        uplift_predicted: np.ndarray,
        treatment_true: Union[pd.Series, np.ndarray],
        with_perfect: bool = True
    ) -> None:
        plt.title('Qini curve')
        if with_perfect:
            curve_perfect = perfect_qini_curve(
                y_true=target_true, treatment=treatment_true
            )
            plt.plot(
                *curve_perfect, **settings.model.plotting_cfg, label='perfect'
            )

        curve_model = qini_curve(
            y_true=target_true,
            uplift=uplift_predicted,
            treatment=treatment_true
        )

        plt.plot(
            *curve_model,
            **settings.model.plotting_cfg,
            label='model'
        )

        plt.plot(
            (curve_model[0][0], curve_model[0][-1]),
            (curve_model[1][0], curve_model[1][-1]),
            **settings.model.plotting_cfg,
            label='random'
        )

        plt.show()
