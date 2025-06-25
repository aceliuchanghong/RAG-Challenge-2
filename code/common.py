from enum import Enum
import re


class DocumentType(Enum):
    markdown = ["md"]
    text = ["txt"]
    word = ["docx"]
    ppt = ["pptx"]
    pdf = ["pdf"]


def extract_hash(file_name):
    pattern = r"^md_([a-zA-Z0-9]+)\.md$"
    match = re.fullmatch(pattern, file_name)
    if match:
        file_hash = match.group(1)
        return file_hash
    else:
        raise ValueError("文件名格式错误，应为 'md_{hash}.md'")
