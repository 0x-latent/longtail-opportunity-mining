"""
简易数据上传服务。

标注者通过浏览器访问上传页面，选择项目名并上传 CSV 文件。
文件自动校验命名规范后上传到腾讯云 COS。

用法:
    pip install flask cos-python-sdk-v5
    python scripts/upload_server.py

    # 自定义端口
    python scripts/upload_server.py --port 8080

    # 允许外网访问（部署到服务器时）
    python scripts/upload_server.py --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

from flask import Flask, jsonify, request

from longtail.config import load_yaml

app = Flask(__name__)

# 加载配置
config = load_yaml(PROJECT_ROOT / "config/default.yaml")
COS_BUCKET = config["cos"]["bucket"]
COS_REGION = config["cos"]["region"]

# 文件命名规则: {project}_{posts|labels}_{YYYYMMDD}.csv
FILENAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_(posts|labels)_\d{8}\.csv$")


def _get_cos_client():
    from qcloud_cos import CosConfig, CosS3Client
    cos_config = CosConfig(
        Region=COS_REGION,
        SecretId=os.environ["COS_SECRET_ID"],
        SecretKey=os.environ["COS_SECRET_KEY"],
    )
    return CosS3Client(cos_config)


UPLOAD_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>数据上传</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; padding: 40px 20px; }
  .container { max-width: 600px; margin: 0 auto; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 32px; font-size: 14px; }
  .card { background: #fff; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 20px; }
  label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 14px; }
  input[type="text"] { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; margin-bottom: 20px; }
  input[type="text"]:focus { outline: none; border-color: #4A90D9; }
  .drop-zone { border: 2px dashed #ccc; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 20px; }
  .drop-zone:hover, .drop-zone.dragover { border-color: #4A90D9; background: #f0f7ff; }
  .drop-zone p { color: #999; font-size: 14px; }
  .drop-zone .icon { font-size: 36px; margin-bottom: 8px; }
  .file-list { margin-bottom: 20px; }
  .file-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; background: #f9f9f9; border-radius: 8px; margin-bottom: 8px; font-size: 13px; }
  .file-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; }
  .file-item .remove { color: #e74c3c; cursor: pointer; margin-left: 12px; font-weight: 600; }
  .btn { width: 100%; padding: 12px; background: #4A90D9; color: #fff; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #357abd; }
  .btn:disabled { background: #ccc; cursor: not-allowed; }
  .msg { padding: 12px; border-radius: 8px; margin-top: 16px; font-size: 14px; display: none; }
  .msg.ok { display: block; background: #e8f5e9; color: #2e7d32; }
  .msg.err { display: block; background: #fbe9e7; color: #c62828; }
  .spec { font-size: 12px; color: #999; margin-top: 8px; line-height: 1.8; }
  .spec code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>
<div class="container">
  <h1>数据上传</h1>
  <p class="subtitle">请按规范上传原始数据和标签表</p>

  <div class="card">
    <label for="project">项目名称</label>
    <input type="text" id="project" placeholder="例如: cold_cough（英文小写+下划线）">

    <label>选择文件</label>
    <div class="drop-zone" id="dropZone">
      <div class="icon">+</div>
      <p>点击或拖拽 CSV 文件到这里</p>
    </div>
    <input type="file" id="fileInput" multiple accept=".csv" style="display:none">

    <div class="file-list" id="fileList"></div>

    <button class="btn" id="uploadBtn" disabled>上传</button>
    <div class="msg" id="msg"></div>

    <div class="spec">
      <strong>命名规范：</strong><br>
      原始数据：<code>{项目名}_posts_{日期}.csv</code>，例如 <code>cold_cough_posts_20260401.csv</code><br>
      标签表：<code>{项目名}_labels_{日期}.csv</code>，例如 <code>cold_cough_labels_20260401.csv</code><br>
      项目名使用英文小写+下划线，日期格式 YYYYMMDD
    </div>
  </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
const msg = document.getElementById('msg');
let files = [];

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => addFiles(fileInput.files));

function addFiles(newFiles) {
  for (const f of newFiles) {
    if (!files.some(x => x.name === f.name)) files.push(f);
  }
  renderFiles();
}

function renderFiles() {
  fileList.innerHTML = files.map((f, i) =>
    `<div class="file-item"><span class="name">${f.name}</span><span class="size">${(f.size/1024).toFixed(1)} KB</span><span class="remove" onclick="removeFile(${i})">x</span></div>`
  ).join('');
  uploadBtn.disabled = files.length === 0;
}

function removeFile(i) { files.splice(i, 1); renderFiles(); }

uploadBtn.addEventListener('click', async () => {
  const project = document.getElementById('project').value.trim();
  if (!project) { showMsg('err', '请填写项目名称'); return; }
  if (!/^[a-z][a-z0-9_]*$/.test(project)) { showMsg('err', '项目名格式错误（英文小写+下划线）'); return; }

  uploadBtn.disabled = true;
  uploadBtn.textContent = '上传中...';
  msg.className = 'msg';

  const formData = new FormData();
  formData.append('project', project);
  files.forEach(f => formData.append('files', f));

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.ok) {
      showMsg('ok', `上传成功！共 ${data.uploaded.length} 个文件`);
      files = []; renderFiles();
    } else {
      showMsg('err', data.error);
    }
  } catch(e) {
    showMsg('err', '上传失败: ' + e.message);
  }
  uploadBtn.disabled = false;
  uploadBtn.textContent = '上传';
});

function showMsg(type, text) { msg.className = 'msg ' + type; msg.textContent = text; }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return UPLOAD_PAGE


@app.route("/upload", methods=["POST"])
def upload():
    project = request.form.get("project", "").strip()
    if not project or not re.match(r"^[a-z][a-z0-9_]*$", project):
        return jsonify(ok=False, error="项目名格式错误（英文小写+下划线）")

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify(ok=False, error="未选择文件")

    # 校验文件名
    for f in uploaded_files:
        if not f.filename.endswith(".csv"):
            return jsonify(ok=False, error=f"文件 {f.filename} 不是 CSV 格式")
        if not FILENAME_PATTERN.match(f.filename):
            return jsonify(
                ok=False,
                error=f"文件 {f.filename} 命名不规范。"
                      f"请使用格式: {project}_posts_YYYYMMDD.csv 或 {project}_labels_YYYYMMDD.csv",
            )
        # 检查文件名中的项目名是否匹配
        if not f.filename.startswith(project + "_"):
            return jsonify(ok=False, error=f"文件 {f.filename} 的项目名与所选项目 '{project}' 不一致")

    # 上传到 COS
    client = _get_cos_client()
    results = []
    for f in uploaded_files:
        cos_key = f"{project}/{f.filename}"
        client.put_object(Bucket=COS_BUCKET, Body=f.stream, Key=cos_key)
        results.append(cos_key)

    return jsonify(ok=True, uploaded=results)


def main():
    parser = argparse.ArgumentParser(description="Data upload server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"  Upload server: http://{args.host}:{args.port}")
    print(f"  COS bucket: {COS_BUCKET} ({COS_REGION})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
