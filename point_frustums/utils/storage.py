"""
Minimal file access that works for local paths and S3-compatible object storage (AWS S3, Cloudflare R2, ...).

Remote paths use the ``s3://bucket/prefix`` scheme. Credentials and the endpoint are taken from the standard AWS
environment variables, which are read by boto3 (and thereby litdata):

    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  # e.g. an R2 API token with object read/write permissions
    AWS_ENDPOINT_URL                          # e.g. https://<account_id>.r2.cloudflarestorage.com (R2)
    AWS_DEFAULT_REGION                        # "auto" for R2
"""

import os
import posixpath
import shutil
import tempfile
from typing import Any, Optional

from litdata.streaming.fs_provider import S3FsProvider


def is_remote(path: str) -> bool:
    return "://" in path


def join(path: str, *parts: str) -> str:
    """Join path components; uses forward slashes for remote paths."""
    if is_remote(path):
        return posixpath.join(path, *parts)
    return os.path.join(path, *parts)


def _provider(path: str, storage_options: Optional[dict[str, Any]] = None) -> S3FsProvider:
    if not path.startswith("s3://"):
        raise ValueError(f"Unsupported remote path '{path}', only s3:// (incl. S3-compatible storage) is supported")
    return S3FsProvider(storage_options=storage_options or {})


def exists(path: str, storage_options: Optional[dict[str, Any]] = None) -> bool:
    if is_remote(path):
        return _provider(path, storage_options).exists(path)
    return os.path.exists(path)


def read_bytes(path: str, storage_options: Optional[dict[str, Any]] = None) -> bytes:
    if not is_remote(path):
        with open(path, "rb") as f:
            return f.read()
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, posixpath.basename(path))
        _provider(path, storage_options).download_file(path, local)
        with open(local, "rb") as f:
            return f.read()


def write_file(local_path: str, path: str, storage_options: Optional[dict[str, Any]] = None) -> None:
    """Copy (local) or upload (remote) the file at local_path to path."""
    if is_remote(path):
        _provider(path, storage_options).upload_file(local_path, path)
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Write to a temporary file first so that readers never see a partially written file
    tmp = f"{path}.tmp"
    shutil.copyfile(local_path, tmp)
    os.replace(tmp, path)
