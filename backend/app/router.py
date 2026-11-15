"""文档解析器路由模块 - 根据文件扩展名选择合适的解析器。

ParserKind 定义了四种解析策略：
    LANGCHAIN: 使用 LangChain 社区加载器处理 CSV/JSON/HTML/DOCX。
    NATIVE: 使用项目自研解析器处理 MD/XLSX/PPTX。
    MINERU: 使用 MinerU 服务处理 PDF/图片。
    AUDIO: 使用语音转文字模型处理音频文件。
"""

from enum import StrEnum
from pathlib import Path


class ParserKind(StrEnum):
    """文档解析策略枚举。"""
    LANGCHAIN = "langchain"
    NATIVE = "native"
    MINERU = "mineru"
    AUDIO = "audio"


# 文件后缀到解析器的映射表
PARSER_BY_SUFFIX = {
    ".csv": ParserKind.LANGCHAIN,
    ".json": ParserKind.LANGCHAIN,
    ".html": ParserKind.LANGCHAIN,
    ".htm": ParserKind.LANGCHAIN,
    ".docx": ParserKind.LANGCHAIN,
    ".md": ParserKind.NATIVE,
    ".markdown": ParserKind.NATIVE,
    ".xlsx": ParserKind.NATIVE,
    ".pptx": ParserKind.NATIVE,
    ".pdf": ParserKind.MINERU,
    ".png": ParserKind.MINERU,
    ".jpg": ParserKind.MINERU,
    ".jpeg": ParserKind.MINERU,
    ".wav": ParserKind.AUDIO,
    ".mp3": ParserKind.AUDIO,
    ".m4a": ParserKind.AUDIO,
}


def choose_parser(filename: str) -> ParserKind:
    """根据文件名后缀选择合适的解析器类型。
    
    Args:
        filename: 原始文件名。
        
    Returns:
        对应的 ParserKind 枚举值。
        
    Raises:
        ValueError: 不支持的文件格式。
    """
    suffix = Path(filename).suffix.lower()
    try:
        return PARSER_BY_SUFFIX[suffix]
    except KeyError as exc:
        raise ValueError(f"不支持的文件格式: {suffix or '无扩展名'}") from exc