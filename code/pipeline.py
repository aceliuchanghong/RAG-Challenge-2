from pathlib import Path
import pypdf
import docx
import pptx
from typing import List, Union

from .common import DocumentType
from .chunking import CustomRecursiveCharacterTextSplitter
from .ingest_with_lancedb import LanceDBIngestor
from .questions_processing import QuestionsProcessor


class Pipeline:
    def __init__(
        self,
        root_path: Path,
        db_path: Union[str, Path] = "./lancedb",
        table_name: str = "file_chunks",
    ):
        """Initialize the pipeline."""
        self.root_path = root_path
        self.db_path = db_path
        self.table_name = table_name
        self.ingestor = None
        self.questions_processor = None

    def read_file(self, file_path: str) -> str:
        """
        Reads a file based on its extension and returns its content as a string.
        Supports file types defined in DocumentType enum.
        """
        path = Path(file_path)
        extension = path.suffix[1:].lower()
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
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
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

    def save2lacncedb(
        self, report_or_reports_dir: str, table_name: str = "file_chunks"
    ):
        """
        处理所有报告，并将数据分批次存入 LanceDB。

        连接到数据库并打开表
        db = lancedb.connect(LANCEDB_PATH)
        table = db.open_table("file_chunks")

        转换为 Pandas DataFrame 查看前 10 条数据
        df_head = table.limit(10).to_pandas()
        print(df_head)
        """
        self.ingestor = LanceDBIngestor(self.db_path)
        self.ingestor.process_and_ingest_reports(report_or_reports_dir, table_name)

    def find_question_related_docs(
        self, question: str, table_name, complicated_question: bool = True
    ) -> List[str]:
        """
        Process a list of questions and return answers.
        """
        self.table_name = table_name
        self.questions_processor = QuestionsProcessor(self.db_path, self.table_name)
        print(f"processing:{question}...")
        question_related_docs = self.questions_processor.find_question_related_docs(
            question, complicated_question
        )

        return question_related_docs
