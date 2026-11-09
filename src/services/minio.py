from __future__ import annotations

from minio import Minio
from minio.error import S3Error
import structlog

from src.common.settings import settings

logger = structlog.get_logger(__name__)

class MinioClient:
    _minio_instance: MinioClient | None = None

    def __init__(self) -> None:
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket_exist()
        logger.debug("Minio instance initialized")

    @classmethod
    def get_instance(cls) -> MinioClient:
        if cls._minio_instance is None:
            cls._minio_instance = cls()
        return cls._minio_instance

    def _ensure_bucket_exist(self) -> None:
        found = self._minio_instance.bucket_exists(bucket_name=self.bucket)
        if not found:
            self._minio_instance.make_bucket(bucket_name=self.bucket)
            logger.info(f"Bucket {self.bucket} created")
        else:
            logger.info(f"Bucket {self.bucket} already exists")

    def _upload_object(self, object_path: str, file_path: str) -> None:
        self._minio_instance.fput_object(
            bucket_name=self.bucket,
            object_name=object_path,
            file_path=file_path
        )
        logger.info(f"Object {object_path} uploaded to MinIO")

    def _download_object(self, object_path: str, file_path: str) -> None:
        try:
            response = self._minio_instance.fget_object(
                bucket_name=self.bucket,
                object_name=object_path,
                file_path=file_path
            )
            logger.info(f"Object {object_path} downloaded from MinIO")
        except Exception as e:
            logger.exception(f"Error getting object {object_path} from Minio: {e}")
            raise
        return response

    def _check_whether_object_exists(self, object_path: str) -> None:
        try:
            self._minio_instance.stat_object(
                bucket_name=self.bucket,
                object_name=object_path
            )
            logger.info(f"Object {object_path} exists in bucket {self.bucket}")
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.info(f"Object {object_path} does not exist in bucket {self.bucket}")
                return False
            logger.error(f"S3 error in MinIO while checking {object_path}: {e}")
            raise
        except Exception as e:
            logger.exception(f"An error ocurred: {e}")
            raise
