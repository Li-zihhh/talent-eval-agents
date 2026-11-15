"""自研文档解析器模块 - 处理 CSV、JSON、XLSX、PPTX 等格式文件。

提供不依赖第三方解析服务的本地解析能力。
"""

import csv
import json
from io import BytesIO, StringIO
from typing import Any

from openpyxl import load_workbook


def parse_csv(content: bytes) -> dict[str, Any]:
    """解析 CSV 文件内容为结构化数据。
    
    Args:
        content: CSV 文件的字节内容。
        
    Returns:
        包含列名和行数据的字典。
    """
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    return {
        "kind": "table",
        "columns": reader.fieldnames or [],
        "rows": list(reader),
    }


def parse_json(content: bytes) -> dict[str, Any]:
    """解析 JSON 文件内容为结构化数据。
    
    Args:
        content: JSON 文件的字节内容。
        
    Returns:
        包含 JSON 数据的字典。
    """
    return {"kind": "json", "data": json.loads(content.decode("utf-8-sig"))}


def parse_xlsx(content: bytes) -> dict[str, Any]:
    """解析 XLSX 文件内容，返回所有工作表的数据。
    
    Args:
        content: XLSX 文件的字节内容。
        
    Returns:
        包含所有 Sheet 名称和行数据的字典。
    """
    workbook = load_workbook(BytesIO(content), data_only=False, read_only=True)
    sheets = [
        {"name": sheet.title, "rows": [list(row) for row in sheet.iter_rows(values_only=True)]}
        for sheet in workbook.worksheets
    ]
    return {"kind": "workbook", "sheets": sheets}


def parse_pptx(file_path: str) -> dict[str, Any]:
    """解析 PPTX 文件，提取每张幻灯片的标题和正文块。
    
    Args:
        file_path: PPTX 文件路径。
        
    Returns:
        包含幻灯片内容列表和数量的字典。
    """
    from pptx import Presentation

    presentation = Presentation(file_path)
    blocks: list[str] = []
    content_list: list[dict[str, Any]] = []
    for number, slide in enumerate(presentation.slides, start=1):
        blocks.append(f"## Slide {number}")
        content_list.append({"type": "title", "text": f"Slide {number}", "locator": {"kind": "slide", "slide": number}})
        for shape_index, shape in enumerate(slide.shapes, start=1):
            text = shape.text.strip() if hasattr(shape, "text") else ""
            if not text:
                continue
            # 如果是幻灯片标题占位符，标记为 title
            is_title = shape == slide.shapes.title
            blocks.append(f"### {text}" if is_title else text)
            content_list.append(
                {
                    "type": "title" if is_title else "text",
                    "text": text,
                    "locator": {
                        "kind": "slide_region",
                        "slide": number,
                        "shape": shape_index,
                        "bbox": [int(shape.left), int(shape.top), int(shape.left + shape.width), int(shape.top + shape.height)],
                        "coordinate_space": "pptx_emu",
                    },
                }
            )
    return {"blocks": blocks, "content_list": content_list, "slide_count": len(presentation.slides)}