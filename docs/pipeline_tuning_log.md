# Pipeline 调优记录

本文档记录 longtail-opportunity-mining 项目从 Phase 1 到 Phase 2 各环节遇到的问题、调整方式和结果。

---

## 1. Phase 1: 数据清洗 (Ingest → Label Match → Masking)

### 1.1 初始状态 (`2d0c4c2`)

Phase 1 骨架已搭好，但 `ingest.py` 功能较弱：
- `load_csv()` 只支持固定列名，不能处理中文列名 CSV
- 没有社媒文本去噪（URL、#话题、@提及、emoji、平台水印都保留在文本中）
- 没有编码自动探测（中文 CSV 常见 GBK/GB18030 编码）
- 没有短文本过滤和去重日志

### 1.2 问题: 原始数据格式不匹配

**现象**: 原始 CSV 列名为 `ID`、`文本`、`标签`、`平台`，而代码期望 `post_id`、`text`。

**调整** (`e33ee2c`):
- 新增 `DEFAULT_COLUMN_MAP` 字典: `{"ID": "post_id", "文本": "text", "标签": "labels_raw", "平台": "platform"}`
- `load_csv()` 支持 `column_map` 参数，自动重命名列
- 支持配置覆盖（`config.ingest.column_map`）

### 1.3 问题: CSV 编码探测

**现象**: 中文社媒数据 CSV 文件编码不确定，可能是 UTF-8、GBK 或 GB18030。

**调整** (`e33ee2c`):
- `load_csv()` 新增编码自动探测：依次尝试 `utf-8-sig → utf-8 → gbk → gb18030`
- 使用 `pd.read_csv(path, nrows=5, encoding=enc)` 试读 5 行来验证编码

### 1.4 问题: 社媒噪声污染文本

**现象**: 原始文本中包含大量社交媒体噪声，例如：
```
晚上咳嗽睡不着怎么办 #宝妈日常 @DOU+小助手
```
这些噪声会干扰后续的标签匹配和聚类。

**调整** (`e33ee2c`):
- 新增 `clean_social_text()` 函数，按顺序清除：
  1. URL (`https?://...`)
  2. 平台水印 (`DOU+小助手`、`小红书号:xxx`)
  3. 话题标签 (`#宝妈日常#`)
  4. @提及 (`@用户名`)
  5. Emoji (Unicode emoticons/symbols)
  6. 残留孤立 `#` 号
- 处理流程变为: `text → clean_social_text → normalize_text(NFKC) → text_clean`

### 1.5 问题: 短文本和重复数据

**调整** (`e33ee2c`):
- `normalize_dataframe()` 新增 `min_text_len` 参数（默认 10），过滤清洗后过短的文本
- `deduplicate_dataframe()` 添加日志输出，记录过滤/去重的行数
- 结果: **过滤 431 行短文本，去重 261 行**，最终 99,905 条帖子

### 1.6 问题: 相对路径导致运行失败

**现象**: 从非项目根目录运行 `python scripts/run_pipeline.py` 时报错 `FileNotFoundError`，因为 `config/default.yaml` 中的路径（如 `data/raw/三九小儿文本-0309.csv`）是相对于项目根目录的。

**调整** (`b2b15bf`, `2b53c72`):
- `pipeline.py` 中新增 `_resolve()` 函数，将配置中的相对路径基于 `project_root`（config 文件的上两级目录）解析为绝对路径
- 第一次修复遗漏了 labels 文件路径，第二次补上

### 1.7 问题: labels_by_dim 出现异常维度列

**现象**: `parse_labels.py` 生成的 `labels_by_dim.csv` 中，列名出现了 "小儿退热口服液"、"脂溢性皮炎"、"停不下来" 等。

**原因**: 原始数据中少量帖子（约 480 条）的标签格式异常——没有按 `品牌认知.产品诉求.咳嗽.咳痰` 的层级格式标注，而是直接写了一个词（如 `脂溢性皮炎`）。`parse_labels.py` 按 `.` 拆分层级，第一层即为 `top_dim`，导致这些词被当作顶层维度。

**状态**: 已识别为原始数据标注质量问题，暂未修复。正常的顶层维度只有 4 个：**场景、品牌认知、需求、方案**。

---

## 2. Phase 2: Embedding + Clustering

### 2.1 初始实现 (`7f23b62`)

Phase 2 完整实现包含：

| 模块 | 功能 |
|------|------|
| `embedding.py` | sentence-transformers 编码，支持 batch、.npy 磁盘缓存 |
| `clustering.py` | UMAP 降维 + HDBSCAN 聚类 + c-TF-IDF 特征词提取 |
| `pipeline.py` | `process_clusters()` 编排全流程 |
| `run_pipeline.py` | `--phase 2` 入口 |

初始参数配置：
```yaml
embedding.model_name: BAAI/bge-small-zh-v1.5   # 中文 embedding 模型
umap: {n_components: 10, n_neighbors: 15, min_dist: 0.0, metric: cosine}
hdbscan: {min_cluster_size: 10, min_samples: 5, cluster_selection_method: eom}
ctfidf: {ngram_range: [1, 3], top_n_terms: 15, min_df: 2}
residual_ratio_threshold: 0.15
```

### 2.2 第一次运行结果

```
total_posts:     99,905
eligible_posts:  99,903  (仅 2 条被过滤)
n_clusters:      738
noise_count:     49,719
noise_ratio:     49.8%
mean_coherence:  0.87
```

### 2.3 问题 1: 过滤几乎没有生效

**现象**: 99,903/99,905 条帖子通过了 `residual_ratio >= 0.15` 过滤，几乎全量参与聚类。

**原因**: 当前标签词表 (`cold_cough_zh_v1.yaml`) 只有 8 个维度、约 30 个标签条目。对于 10 万条社媒帖子来说，覆盖率极低——大部分帖子的文本在标签匹配后残差比例仍然很高（即大部分内容没有被标签覆盖）。

**影响**: 聚类输入数据量接近全量，没有起到"过滤已知信号、只聚类残差"的设计目的。

**后续方向** (待实施):
- 扩充标签词表（增加更多 surface_forms）
- 或者调高阈值（如 0.5），但这需要对数据的 residual_ratio 分布有更清楚的认识

### 2.4 问题 2: 簇数过多（738 个）

**现象**: HDBSCAN 产生了 738 个聚类，远超设计目标的 15-100 个。大量微型簇（size=10）。

**原因**: `min_cluster_size=10` 太小。对于 10 万条数据，10 条帖子即可成簇，导致过度碎片化。

**调整** (`63d8e4e`):
```yaml
# 调整前
hdbscan: {min_cluster_size: 10, min_samples: 5}

# 调整后
hdbscan: {min_cluster_size: 50, min_samples: 10}
```

**结果**: 738 → **195 个簇**，更接近合理范围。

### 2.5 问题 3: c-TF-IDF 被标点和虚词污染

**现象**: 簇的 top terms 被无意义 token 占据：
```
Cluster #496 | top terms: ,, [, ], 了, [ audience ], ...
```

**原因**: `compute_ctfidf()` 的 tokenizer 只过滤了完整的 mask token（如 `[SYMPTOM]`），但没有过滤：
1. 中文标点（`，`、`。`、`！`等）
2. 高频虚词（`的`、`了`、`是`、`在`等）
3. mask token 被分词器拆散后的碎片（`[`、`]`、`audience` 等）

**调整** (`63d8e4e`):
- 新增 `_STOPWORDS` 停用词表，包含：
  - 中英文标点符号
  - 约 60 个中文高频虚词（`的/了/是/在/我/有/和/就/不` 等）
- `_tokenize()` 函数增加停用词过滤

**调整前 top terms**:
```
#496 | ,, [, ], 了, [ audience ], ...
#648 | ,, 支原体, 了, ], [, 阿奇, ...
```

**调整后 top terms**:
```
# 96 | audience, 园, 生病, 送, 感冒, ...
#145 | 支原体, 阿奇, 支原体 肺炎, 肺炎, 多西, ...
#168 | 哮喘, 变异性 哮喘, 变异性, 喘, ...
```

### 2.6 调整前后对比

| 指标 | 第一次运行 | 调整后 | 目标 |
|------|-----------|--------|------|
| 簇数 | 738 | **195** | 15-100 |
| noise ratio | 49.8% | **43.7%** | < 40% |
| mean coherence | 0.87 | **0.84** | > 0.3 |
| top terms 质量 | 标点/虚词为主 | **有意义的医学/生活词汇** | — |

### 2.7 聚类结果概览（调整后 Top 10 大簇）

| 簇 ID | 规模 | coherence | 主题关键词 | 业务解读 |
|--------|------|-----------|-----------|---------|
| #96 | 2985 | 0.805 | 园、生病、送、感冒 | 幼儿园入园后频繁生病 |
| #115 | 2973 | 0.826 | 乳糖、乳糖酶、蒙脱石散 | 乳糖不耐受/腹泻用药 |
| #145 | 2058 | 0.829 | 支原体、阿奇、多西、肺炎 | 支原体肺炎抗生素方案 |
| #122 | 1891 | 0.836 | 开塞露、便秘、乳果糖 | 儿童便秘用药需求 |
| #72 | 1257 | 0.819 | 过敏、奶粉、牛奶、湿疹 | 牛奶过敏/湿疹管理 |
| #154 | 1225 | 0.852 | 支气管炎、支气管肺炎 | 儿童支气管炎治疗 |
| #155 | 1218 | 0.840 | 肺炎、住院、医院 | 肺炎住院治疗经历 |
| #168 | 1166 | 0.842 | 哮喘、变异性哮喘 | 儿童哮喘长期管理 |
| #187 | 1108 | 0.846 | 鼻窦炎、鼻炎、鼻涕 | 鼻窦炎/鼻炎用药 |
| #134 | 1087 | 0.835 | 美林、退烧、度 | 美林退烧场景 |

---

## 3. 已知待优化问题

### 3.1 残差过滤失效

当前标签词表覆盖率太低（仅约 30 个词条），导致 `residual_ratio` 普遍偏高，过滤形同虚设。

**可能方案**:
- 扩充 `cold_cough_zh_v1.yaml` 标签词表（从 labels_flat.csv 的高频 canonical_value 中提取）
- 或基于 residual_ratio 分布的实际分位数来设定阈值

### 3.2 c-TF-IDF 仍有 mask token 维度名残留

top terms 中仍出现 `audience`、`symptom`、`scene`、`form` 等——这些是 mask token `[AUDIENCE]` 等被分词器拆散后的碎片（`[` 和 `]` 已被停用词过滤，但中间的英文词保留了下来）。

**可能方案**:
- 将 mask token 内部的维度名（`audience`, `symptom`, `scene`, `product`, `emotion`, `efficacy`, `form`, `concern`）加入停用词表
- 或改进 tokenizer，将 `[XXX]` 整体识别后再过滤

### 3.3 noise ratio 仍偏高 (43.7%)

接近一半帖子未归入任何簇，超过 40% 的设计目标。

**可能方案**:
- 调小 `min_cluster_size`（在 30-50 之间寻找平衡）
- 调整 UMAP 参数（增大 `n_neighbors` 可以让结构更全局化）
- 当前 noise 帖子可能是内容过于独特/分散的帖子，也可能包含有价值的长尾信号

### 3.4 大簇内容可能过于笼统

最大的簇（#96, 2985 条; #115, 2973 条）规模较大，内部可能混杂了多个子主题。

**可能方案**:
- 对大簇做二次聚类（sub-clustering）
- 或调大 `min_cluster_size` 的同时减小 UMAP 的 `n_components`（从 10 降到 5），使降维更激进

---

## 4. 运行参考

```bash
# Phase 1: 数据清洗
python scripts/run_pipeline.py --phase 1

# Phase 2: 聚类（首次需联网下载模型 ~90MB）
python scripts/run_pipeline.py --phase 2

# 如需使用 Hugging Face 镜像
set HF_ENDPOINT=https://hf-mirror.com
python scripts/run_pipeline.py --phase 2
```

### 输出文件

| 文件 | 阶段 | 说明 |
|------|------|------|
| `data/processed/processed_posts.parquet` | Phase 1 | 清洗+标注后的全量数据 |
| `data/output/phase1_summary.json` | Phase 1 | 处理摘要 |
| `data/processed/embeddings.npy` | Phase 2 | embedding 缓存（约 195MB） |
| `data/processed/cluster_assignments.parquet` | Phase 2 | post_id → cluster_id 映射 |
| `data/output/clusters.json` | Phase 2 | 每个簇的关键词、coherence、标签分布 |
| `data/output/phase2_summary.json` | Phase 2 | 聚类总体指标 |
