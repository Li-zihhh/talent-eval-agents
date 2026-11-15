"""对象存储路径生成模块 - 定义 S3/MinIO 中文件的统一命名规则。

路径格式：tenants/{tenant_id}/documents/{document_id}/versions/{version_no}/{safe_filename}
"""

import re
from pathlib import PurePath


def build_object_key(tenant_id: str, document_id: str, version_no: int, filename: str) -> str:
    """构建 S3 对象存储中的统一 Key。
    
    对原始文件名做安全清洗，替换非法字符为下划线，确保多租户环境下路径唯一。
    
    Args:
        tenant_id: 租户标识。
        document_id: 文档 UUID。
        version_no: 版本号。
        filename: 原始文件名。
        
    Returns:
        格式化的 S3 Object Key。
    """
    safe_name = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", PurePath(filename).name)
    return f"tenants/{tenant_id}/documents/{document_id}/versions/{version_no}/{safe_name}"