from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key: str
    aemet_station_idema: int

    minio_endpoint: str
    minio_bucket: str
    minio_access_key: str
    minio_secret_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
