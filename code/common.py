from enum import Enum


class DocumentType(Enum):
    markdown = ["md"]
    text = ["txt"]
    word = ["docx"]
    ppt = ["pptx"]
    pdf = ["pdf"]
