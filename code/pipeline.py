from pathlib import Path
import pypdf
import docx
import pptx

from .common import DocumentType
from code.chunking import CustomRecursiveCharacterTextSplitter


class Pipeline:
    def __init__(self, root_path: Path):
        """Initialize the pipeline."""
        self.root_path = root_path

    def read_file(self, file_path: str) -> str:
        """
        Reads a file based on its extension and returns its content as a string.
        Supports file types defined in DocumentType enum.
        """
        path = Path(file_path)
        extension = path.suffix[1:].lower()  # 获取后缀名并转为小写, 如: "pdf"

        content = ""

        if extension in DocumentType.pdf.value:
            # 处理 PDF 文件
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text() or ""

        elif extension in DocumentType.word.value:
            # 处理 DOCX 文件
            doc = docx.Document(path)
            for para in doc.paragraphs:
                content += para.text + "\n"

        elif extension in DocumentType.ppt.value:
            # 处理 PPTX 文件
            prs = pptx.Presentation(path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        content += shape.text + "\n"

        elif (
            extension in DocumentType.markdown.value
            or extension in DocumentType.text.value
        ):
            # 处理 Markdown 和纯文本文件
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        else:
            # 如果文件类型不支持，则抛出异常
            supported_types = [
                ext for doc_type in DocumentType for ext in doc_type.value
            ]
            raise ValueError(
                f"Unsupported file type: '.{extension}'. Supported types are: {supported_types}"
            )

        return content

    def chunk_md_file(
        self, md_file_path: str, chunk_size: int = 300, chunk_overlap: int = 50
    ) -> dict:
        """
        Chunk a markdown file into smaller parts based on the specified chunk size and overlap.
        """

        splitter = CustomRecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(md_file_path)
