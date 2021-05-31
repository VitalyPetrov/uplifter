from typing import List, Dict, Any
from pydantic import BaseSettings, Field


class DataProcessorSettings(BaseSettings):
    treatment_nm: str = Field(default="treatment_flg")
    target_nm: str = Field(default="target_flg")
    test_ratio: float = Field(gt=0.0, lt=1.0, default=0.1)


class ModelSettings(BaseSettings):
    num_trees: int = 1000
    thread_count: int = 8
    cat_features: List[str] = []

    plotting_cfg: Dict[str, Any] = {"lw": 4.5, "ls": "solid"}


class Settings:
    data: DataProcessorSettings = DataProcessorSettings()
    model: ModelSettings = ModelSettings()


settings = Settings()
