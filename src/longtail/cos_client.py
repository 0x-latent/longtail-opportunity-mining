"""腾讯云 COS 数据同步模块。

从 COS Bucket 下载项目数据到本地 data/raw/ 目录。
需要配置环境变量或 .env 文件：COS_SECRET_ID, COS_SECRET_KEY
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_client(secret_id: str, secret_key: str, region: str):
    """创建 COS 客户端。"""
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError as exc:
        raise RuntimeError(
            "缺少 cos-python-sdk-v5，请运行: pip install cos-python-sdk-v5"
        ) from exc

    config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
    return CosS3Client(config)


def list_project_files(
    bucket: str,
    region: str,
    project: str,
    secret_id: str | None = None,
    secret_key: str | None = None,
) -> list[str]:
    """列出 COS 上指定项目目录下的文件。"""
    secret_id = secret_id or os.environ["COS_SECRET_ID"]
    secret_key = secret_key or os.environ["COS_SECRET_KEY"]
    client = _get_client(secret_id, secret_key, region)

    prefix = f"{project}/"
    response = client.list_objects(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    return [obj["Key"] for obj in contents if not obj["Key"].endswith("/")]


def download_project_data(
    bucket: str,
    region: str,
    project: str,
    local_dir: str | Path,
    secret_id: str | None = None,
    secret_key: str | None = None,
) -> dict[str, Path]:
    """
    从 COS 下载项目数据到本地目录。

    COS 目录结构：
        {project}/{project}_posts_{date}.csv
        {project}/{project}_labels_{date}.csv

    返回 {"posts": local_path, "labels": local_path} (最新日期的文件)
    """
    secret_id = secret_id or os.environ["COS_SECRET_ID"]
    secret_key = secret_key or os.environ["COS_SECRET_KEY"]
    client = _get_client(secret_id, secret_key, region)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # 列出项目目录下所有文件
    files = list_project_files(bucket, region, project, secret_id, secret_key)
    if not files:
        raise FileNotFoundError(f"COS 上未找到项目 '{project}' 的数据 (bucket={bucket}, prefix={project}/)")

    # 按类型分组，取最新日期
    posts_files = sorted([f for f in files if "_posts_" in f])
    labels_files = sorted([f for f in files if "_labels_" in f])

    downloaded = {}

    for file_type, file_list in [("posts", posts_files), ("labels", labels_files)]:
        if not file_list:
            logger.warning("未找到 %s 类型的文件", file_type)
            continue
        # 取最后一个（按文件名排序，日期最新的在最后）
        cos_key = file_list[-1]
        filename = cos_key.split("/")[-1]
        local_path = local_dir / filename

        # 如果本地已存在且大小一致，跳过下载
        if local_path.exists():
            response = client.head_object(Bucket=bucket, Key=cos_key)
            remote_size = int(response["Content-Length"])
            if local_path.stat().st_size == remote_size:
                logger.info("跳过下载 (本地已存在): %s", filename)
                downloaded[file_type] = local_path
                continue

        logger.info("下载: cos://%s/%s → %s", bucket, cos_key, local_path)
        client.download_file(Bucket=bucket, Key=cos_key, DestFilePath=str(local_path))
        downloaded[file_type] = local_path

    return downloaded


def sync_project(cos_config: dict[str, Any], project: str, local_dir: str | Path) -> dict[str, Path]:
    """
    根据配置同步项目数据。

    cos_config 格式 (from default.yaml):
        bucket: social-listening-1307163315
        region: ap-guangzhou
    """
    return download_project_data(
        bucket=cos_config["bucket"],
        region=cos_config["region"],
        project=project,
        local_dir=local_dir,
    )
