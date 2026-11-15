"""对象存储客户端模块 - 封装 MinIO / S3 兼容存储的读写操作。

提供统一的文件上传、下载、预签名 URL 生成和删除功能。
"""

import logging
from io import BytesIO

import boto3
from botocore.client import Config

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)


class ObjectStore:
    """基于 boto3 的 S3 兼容对象存储客户端。
    
    同时维护内网和公网两个 client：内网用于数据操作，公网用于生成预签名 URL。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        # 内网 client，用于上传/下载/删除
        self.client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        # 公网 client，用于生成对外可访问的预签名 URL
        self.public_client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_public_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        """确保 Bucket 存在，若不存在则自动创建。"""
        try:
            self.client.head_bucket(Bucket=self.settings.s3_bucket)
            logger.info("object_store_bucket_ready bucket=%s", self.settings.s3_bucket)
        except Exception as exc:
            logger.warning("object_store_bucket_missing bucket=%s reason=%s", self.settings.s3_bucket, exc)
            self.client.create_bucket(Bucket=self.settings.s3_bucket)
            logger.info("object_store_bucket_created bucket=%s", self.settings.s3_bucket)

    def put_bytes(self, key: str, content: bytes, content_type: str) -> None:
        """上传字节内容到指定 Key。
        
        Args:
            key: S3 Object Key。
            content: 文件内容字节。
            content_type: MIME 类型。
        """
        logger.info(
            "object_store_put_started bucket=%s key=%s size_bytes=%s content_type=%s",
            self.settings.s3_bucket,
            key,
            len(content),
            content_type,
        )
        self.client.upload_fileobj(BytesIO(content), self.settings.s3_bucket, key, ExtraArgs={"ContentType": content_type})
        logger.info("object_store_put_completed bucket=%s key=%s", self.settings.s3_bucket, key)

    def get_bytes(self, key: str) -> bytes:
        """从指定 Key 下载文件内容。
        
        Args:
            key: S3 Object Key。
            
        Returns:
            文件内容的字节数据。
        """
        logger.info("object_store_get_started bucket=%s key=%s", self.settings.s3_bucket, key)
        content = self.client.get_object(Bucket=self.settings.s3_bucket, Key=key)["Body"].read()
        logger.info("object_store_get_completed bucket=%s key=%s size_bytes=%s", self.settings.s3_bucket, key, len(content))
        return content

    def presigned_get(self, key: str, expires: int = 900) -> str:
        """生成带有效期限制的预签名下载 URL。
        
        Args:
            key: S3 Object Key。
            expires: 过期时间（秒），默认 15 分钟。
            
        Returns:
            可在公网直接访问的临时下载链接。
        """
        url = self.public_client.generate_presigned_url("get_object", Params={"Bucket": self.settings.s3_bucket, "Key": key}, ExpiresIn=expires)
        logger.info("object_store_presigned_url_created bucket=%s key=%s expires_seconds=%s", self.settings.s3_bucket, key, expires)
        return url

    def delete(self, key: str) -> None:
        """删除指定 Key 的对象。
        
        Args:
            key: S3 Object Key。
        """
        logger.info("object_store_delete_started bucket=%s key=%s", self.settings.s3_bucket, key)
        self.client.delete_object(Bucket=self.settings.s3_bucket, Key=key)
        logger.info("object_store_delete_completed bucket=%s key=%s", self.settings.s3_bucket, key)