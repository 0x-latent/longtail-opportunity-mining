"""共享工具函数"""
import re
import unicodedata


MASK_PATTERN = re.compile(r"\[[A-Z_]+\]")


def normalize_text(text: str) -> str:
    """中文文本标准化：去除多余空白、全角转半角、统一标点"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_chinese(text: str) -> list[str]:
    """保留 mask token，再对其余中文片段做轻量切分。"""
    if not text:
        return []

    tokens: list[str] = []
    last = 0
    try:
        import jieba

        def segment(chunk: str) -> list[str]:
            return [t for t in jieba.cut(chunk, cut_all=False) if t and not t.isspace()]
    except ImportError:
        def segment(chunk: str) -> list[str]:
            return [ch for ch in chunk if not ch.isspace()]

    for match in MASK_PATTERN.finditer(text):
        prefix = text[last:match.start()]
        tokens.extend(segment(prefix))
        tokens.append(match.group(0))
        last = match.end()
    tokens.extend(segment(text[last:]))
    return [t for t in tokens if t]


def residual_ratio(text_masked: str, mask_tokens: list[str]) -> float:
    """计算 (非掩码 token 数) / (总 token 数)。"""
    tokens = tokenize_chinese(text_masked)
    if not tokens:
        return 0.0
    mask_count = sum(1 for t in tokens if t in mask_tokens)
    return max(0.0, (len(tokens) - mask_count) / len(tokens))
