import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from src.conf import settings


class DataPool:
    def __init__(
        self,
        dataset: pd.DataFrame,
        treatment_nm: str = settings.data.treatment_nm,
        target_nm: str = settings.data.target_nm,
        test_ratio: float = settings.data.test_ratio,
    ):
        """
        Provide processed dataset for further uplift model serving
        and data analysis

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

        self.train, self.test, self.valid = self._split_interactions()

    def _split_interactions(self) -> List[pd.DataFrame]:
        train, test = train_test_split(
            self.dataset,
            test_size=self.test_ratio,
            stratify=self.dataset[self.target_nm],
        )

        train, valid = train_test_split(
            train, test_size=self.test_ratio, stratify=train[self.target_nm]
        )

        return train, test, valid

    def _extract_features_and_target(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        return data.drop(self.target_nm, axis=1), data[self.target_nm]

    def get_train_data(self):
        return self._extract_features_and_target(self.train)

    def get_test_data(self):
        return self._extract_features_and_target(self.test)

    def get_validation_data(self):
        return self._extract_features_and_target(self.valid)

    def get_treatment_target_distribution(self) -> pd.DataFrame:
        return (
            self.dataset.groupby([self.treatment_nm, self.target_nm])
            .size()
            .reset_index(name="num_customers")
        )

    def plot_treatment_target_distribution(self) -> None:
        sns.countplot(
            self.dataset[self.treatment_nm].astype(str)
            + self.dataset[self.target_nm].astype()
        )

        plt.xlabel("Treatment-target combination")
        plt.ylabel("Number of customers")
