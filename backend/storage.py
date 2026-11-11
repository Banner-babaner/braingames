import os
import io
from minio import Minio

_client = None
BUCKET = os.getenv("MINIO_BUCKET", "documents")

def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
    return _client

def ensure_bucket():
    client = get_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

def upload_file(object_name: str, data: bytes, content_type: str = "application/octet-stream"):
    ensure_bucket()
    get_client().put_object(
        BUCKET, object_name, io.BytesIO(data), length=len(data),
        content_type=content_type
    )