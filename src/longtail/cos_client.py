"""腾讯云 COS 数据同步模块。

从 COS Bucket 下载项目数据到本地 data/raw/ 目录。
需要配置环境变量或 .env 文件：COS_SECRET_ID, COS_SECRET_KEY

COS 目录结构：
    {vendor}/{project}_posts_{date}.csv
    {vendor}/{project}_labels_{date}.csv

一个供应商（vendor）目录下可以有多个项目。
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 从文件名提取项目名和类型：{project}_{type}_{date}.csv
_RE_FILENAME = re.compile(r"^(.+?)_(posts|labels)_(\d{8})\.csv$")


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


def list_vendor_files(
    bucket: str,
    region: str,
    vendor: str,
    secret_id: str | None = None,
    secret_key: str | None = None,
) -> list[str]:
    """列出 COS 上指定供应商目录下的文件。"""
    secret_id = secret_id or os.environ["COS_SECRET_ID"]
    secret_key = secret_key or os.environ["COS_SECRET_KEY"]
    client = _get_client(secret_id, secret_key, region)

    prefix = f"{vendor}/"
    response = client.list_objects(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])
    return [obj["Key"] for obj in contents if not obj["Key"].endswith("/")]


def discover_projects(
    bucket: str,
    region: str,
    vendor: str,
    secret_id: str | None = None,
    secret_key: str | None = None,
) -> dict[str, dict[str, str]]:
    """
    扫描供应商目录，发现所有项目及其最新文件。

    返回:
        {
            "baby_medication_experience": {
                "posts": "hongyuan/baby_medication_experience_posts_20260403.csv",
                "labels": "hongyuan/baby_medication_experience_labels_20260403.csv",
            },
            ...
        }
    """
    files = list_vendor_files(bucket, region, vendor, secret_id, secret_key)

    # 按项目名分组
    projects: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for cos_key in files:
        filename = cos_key.split("/")[-1]
        m = _RE_FILENAME.match(filename)
        if not m:
            logger.warning("跳过无法识别的文件: %s", cos_key)
            continue
        project_name, file_type, _date = m.groups()
        projects[project_name][file_type].append(cos_key)

    # 每个项目每种类型取最新（按文件名排序，日期在最后）
    result: dict[str, dict[str, str]] = {}
    for project_name, type_files in sorted(projects.items()):
        result[project_name] = {}
        for file_type in ("posts", "labels"):
            if file_type in type_files:
                result[project_name][file_type] = sorted(type_files[file_type])[-1]
    return result


def download_project_data(
    bucket: str,
    region: str,
    vendor: str,
    project_name: str,
    project_files: dict[str, str],
    local_dir: str | Path,
    secret_id: str | None = None,
    secret_key: str | None = None,
) -> dict[str, Path]:
    """
    从 COS 下载指定项目的数据到本地目录。

    project_files: discover_projects() 返回的单个项目 dict，如:
        {"posts": "hongyuan/xxx_posts_20260403.csv", "labels": "hongyuan/xxx_labels_20260403.csv"}
    """
    secret_id = secret_id or os.environ["COS_SECRET_ID"]
    secret_key = secret_key or os.environ["COS_SECRET_KEY"]
    client = _get_client(secret_id, secret_key, region)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for file_type, cos_key in project_files.items():
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


def sync_vendor(
    cos_config: dict[str, Any],
    vendor: str,
    local_dir: str | Path,
    project_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    同步一个供应商下的项目数据。

    Args:
        cos_config: {"bucket": ..., "region": ...}
        vendor: 供应商名（COS 文件夹名）
        local_dir: 本地 data/raw/ 目录
        project_filter: 只同步指定项目名（None = 全部）

    Returns:
        [{"project": "baby_medication_experience", "vendor": "hongyuan",
          "files": {"posts": Path, "labels": Path}}, ...]
    """
    bucket = cos_config["bucket"]
    region = cos_config["region"]

    projects = discover_projects(bucket, region, vendor)
    if not projects:
        raise FileNotFoundError(f"COS 上供应商 '{vendor}' 目录下未找到任何项目数据 (bucket={bucket})")

    if project_filter:
        if project_filter not in projects:
            available = ", ".join(projects.keys())
            raise ValueError(f"供应商 '{vendor}' 下未找到项目 '{project_filter}'，可用项目: {available}")
        projects = {project_filter: projects[project_filter]}

    results = []
    for project_name, project_files in projects.items():
        print(f"  同步项目: {vendor}/{project_name}")
        downloaded = download_project_data(
            bucket=bucket,
            region=region,
            vendor=vendor,
            project_name=project_name,
            project_files=project_files,
            local_dir=local_dir,
        )
        results.append({
            "project": project_name,
            "vendor": vendor,
            "files": downloaded,
        })

    return results
