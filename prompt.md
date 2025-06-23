提示词

```python

```

---

我在写rag的一个读取文件的函数,然后通过执行:
`uv run run.py read-files --file-path no_git_oic/test_files/流式细胞制备方案.pdf`
只读`DocumentType`运行的文件,帮我修改完成下面函数

- run.py
```python
import click
import os
from termcolor import colored
from pathlib import Path
from code.pipeline import Pipeline
from z_utils.hash_x import compute_mdhash_id
@click.group()
def cli():
    pass
@cli.command()
@click.option("--file-path", help="file_path")
@click.option("--output", default="output/md", help="file output path")
def read_files(file_path, output):
    pipeline = Pipeline(root_path)
    content = pipeline.read_file(file_path)
    output_name = compute_mdhash_id(content, prefix="md_")
    output_path = os.path.join(output, output_name + ".md")
    if not os.path.exists(output):
        os.makedirs(output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    click.echo(colored(f"File {file_path} read and saved as {output_path}.", "green"))
if __name__ == "__main__":
    root_path = Path.cwd()
    cli()
```
- code/pipeline.py
```python
from pathlib import Path

class Pipeline:
    def __init__(self, root_path: Path):
        """Initialize the pipeline."""
        pass
    def read_file(self, file_path: str) -> str:
        """Read a file and return its content as a string."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
```
- code/common.py
```python
from enum import Enum

class DocumentType(Enum):

    markdown = ["md"]
    text = ["txt"]
    word = ["docx"]
    ppt = ["pptx"]
    pdf = ["pdf"]
```



---

参考下面完善函数,我想做`RecursiveCharacterTextSplitter`,但是不想依赖langchain,
但是tiktoken是可以的,我在考虑用spicy分句子之类的
```python
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables."""
        chunks = []
        chunk_id = 0
        
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
        
        for page in file_content['content']['pages']:
            page_chunks = self._split_page(page)
            for chunk in page_chunks:
                chunk['id'] = chunk_id
                chunk['type'] = 'content'
                chunk_id += 1
                chunks.append(chunk)
            
            if tables_by_page and page['page'] in tables_by_page:
                for table in tables_by_page[page['page']]:
                    table['id'] = chunk_id
                    table['type'] = 'serialized_table'
                    chunk_id += 1
                    chunks.append(table)
        
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):

        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in all_report_paths:
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(f"Warning: Could not find serialized tables report for {report_path.name}")
                
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
                
            updated_report = self._split_report(report_data, serialized_tables_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)
                
        print(f"Split {len(all_report_paths)} files")
```


---




---




---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---



---




---
