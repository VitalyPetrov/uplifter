import pandas as pd
from typing import Dict, List, Tuple, Mapping
from sklearn.model_selection import train_test_split

from src.conf import settings


class DataPool:
    def __init__(
        self,
        dataset: pd.DataFrame,
        treatment_nm: str = settings.data.treatment_nm,
        target_nm: str = settings.data.target_nm,
        test_ratio: float = settings.data.test_ratio
    ):
        """
        Provide processed dataset for further uplift model serving

        Parameters
        ----------
        dataset: pd.DataFrame
            initial dataset containing features, treatment and target flags
        treatment_nm: str
            column name for treatment flag on dataset
        target_nm: str
            column name for target flag on dataset
        test_ratio: float
            ratio of test subset to evaluate model on
        """
        self.dataset = dataset
        self.treatment_nm, self.target_nm = treatment_nm, target_nm
        self.test_ratio = test_ratio


