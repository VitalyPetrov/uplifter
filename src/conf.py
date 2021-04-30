from pydantic import (
    BaseSettings, Field
)


class DataProcessorSettings(BaseSettings):
    treatment_nm: str = Field(default='treatment_flg')
    target_nm: str = Field(default='target_flg')
    test_ratio: float = Field(gt=0.0, lt=1.0, default=0.1)


class Settings:
    data: DataProcessorSettings = DataProcessorSettings()


settings = Settings()
