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

1. lancedb存储之后数据如下,我怎么迁移呢?
2. `ingestor = LanceDBIngestor(db_path=LANCEDB_PATH)`每次都需要新创建一个表吗?
3. 如果所有数据都存一个表里面,会不会不太好?
4. 插入数据时,不能存在了就最多更新,而不是插入一条新的记录吗
5. 我怎么可视化看的见数据存了那些数据呢?我允许了很多遍相同代码,我担心数据重复冗余太多

```
lancedb/
|
└── file_chunks.lance/
    ├── _indices/
    │   ├── 05b0dbcf-b3d6-419c-bbab-7b4919d47b08/
    │   │   ├── metadata.lance
    │   │   ├── part_0_docs.lance
    │   │   ├── part_0_invert.lance
    │   │   └── part_0_tokens.lance
    │       ├── metadata.lance
    │       ├── part_6_docs.lance
    │       ├── part_6_invert.lance
    │       └── part_6_tokens.lance
    ├── _transactions/
    │   ├── 0-b7550b92-de81-4e4f-9934-070762d3a22f.txn
    │   └── 9-e9e1e515-235b-43dd-83c6-85fb1e83ed2f.txn
    ├── _versions/
    │   ├── 1.manifest
    │   ├── 8.manifest
    │   └── 9.manifest
    └── data/
        ├── 2aa4738a-0cba-4739-9977-f45e515b5a15.lance
        ├── 33b9a896-fead-4b28-ae8c-dffd4dcf9be9.lance
```

```python
class LanceDBIngestor:
    def __init__(self, db_path: Union[str, Path] = "./lancedb"):
        load_dotenv()
        self.llm = OpenAI(
            api_key=os.getenv("EMB_API_KEY"),
            base_url=os.getenv("EMB_BASE_URL"),
            max_retries=2,
        )
        self.db = lancedb.connect(db_path)
        self.table = self.db.create_table(
            "file_chunks", schema=LanceDBSchema, mode="overwrite"
        )
    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
    def _get_embeddings(
        self, texts: List[str], model: str = "Qwen3-Embedding-4B"
    ) -> List[List[float]]:
        if not texts:
            return []
        texts = [t.replace("\n", " ") for t in texts if t.strip()]
        if not texts:
            return []
        response = self.llm.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]
    def process_and_ingest_reports(self, all_reports_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.jsonl"))
        print(f"{all_report_paths}")
        all_data_to_add = []
        for report_path in tqdm(all_report_paths, desc="[1/3] 解析报告并生成向量"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            sha1_name = report_data["file_hash"]
            text_chunks = [chunk["content"] for chunk in report_data["chunks"]]
            page_nums = [chunk["page_num"] for chunk in report_data["chunks"]]
            embeddings = self._get_embeddings(text_chunks)
            if len(text_chunks) != len(embeddings):
                print(f"警告: 报告 {sha1_name} 的文本块和向量数量不匹配。跳过此报告。")
                continue
            for _, (i, text, vector) in enumerate(
                zip(page_nums, text_chunks, embeddings)
            ):
                segmented_text = " ".join(jieba.cut_for_search(text))
                all_data_to_add.append(
                    {
                        "text": text,
                        "text_for_fts": segmented_text,
                        "vector": vector,
                        "report_sha1": sha1_name,
                        "chunk_id": i,
                    }
                )
        if all_data_to_add:
            print(f"\n[2/3] 正在向 LanceDB 表中添加 {len(all_data_to_add)} 个数据块...")
            self.table.add(all_data_to_add)
            print("数据添加完成。")
        print("[3/3] 正在创建全文搜索 (FTS) 索引...")
        self.table.create_fts_index("text_for_fts", replace=True)
        print("FTS 索引创建完成。")
        print(f"\n处理了 {len(all_report_paths)} 个报告，数据库构建完成！")
    def keyword_search(self, query: str, limit: int = 2):
        print(f"\n--- 关键字搜索: '{query}' ---")
        segmented_query = " ".join(jieba.cut_for_search(query))
        results = (
            self.table.search(segmented_query).limit(limit).to_pydantic(LanceDBSchema)
        )
        for res in results:
            print(f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}]")
            print(f"    文本: {res.text[:150]}...\n")
        return results
    def vector_search(self, query: str, limit: int = 1):
        print(f"\n--- 向量搜索: '{query}' ---")
        query_vector = self._get_embeddings([query])[0]
        results = (
            self.table.search(query_vector).limit(limit).to_pydantic(LanceDBSchema)
        )
        for res in results:
            print(f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}]")
            print(f"    文本: {res.text[:150]}...\n")
```

---

`lancedb/` 目录结构本身就是完整的数据库。这是 LanceDB 作为一个嵌入式、无服务器数据库的核心优势之一。
**迁移方法非常简单：直接复制整个 `lancedb` 目录即可。**

可以通过将 LanceDB 表转换为常见的 Python 数据分析库（如 Pandas）的格式来轻松查看和分析数据。

```python
import pandas as pd
import lancedb

# 连接数据库
db = lancedb.connect("./lancedb")  # 使用您的数据库路径
table = db.open_table("file_chunks")

# 将整个表加载到 Pandas DataFrame
# 如果表非常大，这可能会消耗很多内存。
print("正在加载数据到 Pandas... (如果数据量大，可能需要一些时间)")
df = table.to_pandas()
print("数据加载完成。")

# 定义唯一标识符列
primary_keys = ["report_sha1", "chunk_id"]

# 查找重复的行
# keep=False 会标记所有重复项，而不仅仅是第二个及以后的
duplicates = df[df.duplicated(subset=primary_keys, keep=False)]

if not duplicates.empty:
    print("\n!!! 发现重复记录 !!!")
    # 按主键排序，以便更容易地看到重复组
    print(duplicates.sort_values(by=primary_keys))
else:
    print("\n恭喜！未在表中发现基于 (report_sha1, chunk_id) 的重复记录。")

# 也可以查看重复项的统计信息
print(f"\n总行数: {len(df)}")
print(f"重复行数: {len(duplicates)}")
print(
    f"唯一 (report_sha1, chunk_id) 组合的数量: {len(df.drop_duplicates(subset=primary_keys))}"
)
```

---

还有个问题,正如上面看见的`lancedb`的目录,似乎每次运行,该目录就会增加很多文件


LanceDB 和许多现代数据系统（如 Delta Lake）一样，其核心设计是基于**不可变数据文件**的。这意味着：

1.  **数据文件一旦写入，就不会被修改。** 当您向表中添加或更新数据时，LanceDB 不会去改变现有的数据文件。相反，它会写入包含新数据或更新后数据的**新文件**。
2.  **所有操作都是通过版本控制来管理的。** 每一次成功的写入操作（如 `add`, `create_table`, `create_fts_index`）都会创建一个新的**版本**。

这个机制被称为 多版本并发控制 (MVCC)。现在我们来结合您看到的目录结构解释这个过程：

  * `data/`: 这个目录存放着实际的数据块（`.lance` 文件）。当您调用 `table.add()` 时，新的数据会被写入一个新的或多个新的 `.lance` 文件中。
  * `_versions/`: 这是版本控制的核心。每次您成功提交一个事务（比如一次数据添加），这里就会生成一个新的清单文件（`X.manifest`）。这个清单文件是一个小小的元数据文件，它记录了构成当前表“版本X”的所有数据文件是 `data/` 目录下的哪些文件。
  * `_transactions/`: 用于确保操作的原子性。在写入新版本之前，会先创建一个事务文件。操作成功后，事务被提交，并生成新的 `.manifest` 文件。这可以防止数据库在写入过程中因意外中断而损坏。
  * `_indices/`: 当您创建或更新索引时（例如 `create_fts_index`），索引本身的数据也会被写入这个目录下的新文件中。


---

lancedb如何回退某个表的版本呢?

```python
import lancedb
import pandas as pd

db = lancedb.connect("./my_db")
table = db.open_table("my_table")
versions = table.list_versions()

# 恢复到版本 1
table.restore(1)
# checkout 到该版本
table.checkout(1)
```

| 功能 | `checkout(version)` | `restore(version)` |
| :--- | :--- | :--- |
| **操作性质** | 只读（临时切换） | 写（永久恢复） |
| **是否创建新版本** | 否 | 是 |
| **主要用途** | 数据查看、分析、调试 | 数据恢复、错误修正 |
| **对后续操作的影响** | 不影响表的最新状态，可随时 `checkout_latest()` | 创建一个新的“最新”版本，后续写入将在此基础上进行 |

---

```shell
uv run run.py read-files --file-path no_git_oic/material_prediction_files --output no_git_oic/mp_output/md
uv run run.py chunk-markdown --md-file-path no_git_oic/mp_output/md --output no_git_oic/mp_output/chunked_md
uv run run.py save-jsonl --jsonl-path-or-dir no_git_oic/mp_output/chunked_md --table-name mp_database
uv run run.py get-docs --question "CaCO3的A位、B位掺杂,有哪些比较合适?" --table-name mp_database --complicated-question --tags test1
uv run run.py answer-question --table-name mp_database --question "CaCO3的A位、B位掺杂,有哪些比较合适?" --stream --complicated-question
```

---

```shell
uv run run.py read-files --file-path no_git_oic/material_prediction_files2 --output no_git_oic/mp_output2/md
uv run run.py chunk-markdown --md-file-path no_git_oic/mp_output2/md --output no_git_oic/mp_output2/chunked_md
uv run run.py save-jsonl --jsonl-path-or-dir no_git_oic/mp_output2/chunked_md --table-name mp_database
```

```shell
uv run run.py read-files --file-path no_git_oic/material_prediction_files3 --output no_git_oic/mp_output3/md
uv run run.py chunk-markdown --md-file-path no_git_oic/mp_output3/md --output no_git_oic/mp_output3/chunked_md
uv run run.py save-jsonl --jsonl-path-or-dir no_git_oic/mp_output3/chunked_md --table-name mp_database
```

---

```shell
uv run run.py read-files --file-path no_git_oic/torch_test_files --output no_git_oic/torch_output/md
uv run run.py chunk-markdown --md-file-path no_git_oic/torch_output/md --output no_git_oic/torch_output/chunked_md
uv run run.py save-jsonl --jsonl-path-or-dir no_git_oic/torch_output/chunked_md --table-name torch_database
uv run run.py answer-question --table-name torch_database --question "上班如果迟到怎么处理" --stream --complicated-question
```

```shell
uv run run.py read-files --file-path 'deal_minerU/no_git_oic/upload_files_md_output/layout2 copy/vlm/layout2_copy.md' --output no_git_oic/read_test_dir/md
uv run run.py chunk-markdown --md-file-path no_git_oic/read_test_dir/md --output no_git_oic/read_test_dir/chunked_md
```

```shell
uv run run.py read-files --file-path 'deal_minerU/no_git_oic/save_md/NPD2308.md' --output no_git_oic/mlcc/md
uv run run.py chunk-markdown --md-file-path no_git_oic/mlcc/md --output no_git_oic/mlcc/chunked_md --tags CT4701 --ser-tab
uv run run.py save-jsonl --jsonl-path-or-dir no_git_oic/mlcc/chunked_md --table-name mlcc_database
uv run run.py answer-question --table-name mlcc_database --question "CT4701型金属支架表面贴装脉冲功率瓷介固定电容器框架电镀工序工艺规程解释" --stream --tags CT4701

# 工艺流程卡(TE-QR-G914-1)里面锡铅槽成分的控制要求
# CT4701型金属支架表面贴装脉冲功率瓷介固定电容器框架电镀工序工艺规程解释
# 工序号为G02的固化温度和固化时间
```

---

我在做rag,设计向量数据库的表,我在考虑要不要加2个备用字段和一个tags字段,这个tags字段怎么设计呢?
```python
class LanceDBSchema(LanceModel):
    """定义数据模型 (Schema)"""

    text: Optional[str]  # 原始文本块内容
    text_for_fts: Optional[str]  # 分词后的文本，用于FTS索引
    vector: Vector(2560)  # 文本对应的向量
    report_sha1: Optional[str]  # 该文本块所属报告的SHA1标识符
    chunk_id: Optional[int]  # 文本块在原报告中的索引位置
```

---

```shell
curl -X 'POST' \
  'http://127.0.0.1:5000/get-docs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "CaTiO3解释?",
    "table_name": "mp_database",
    "complicated_question": false,
    "tags": ["python"]
  }'
```


---

cd no_git_oic/OCRFlux
python -m ocrflux.pipeline ./localworkspace --data /mnt/data/llch/my_lm_log/no_git_oic/images/C23206C4_page_1.png --model /mnt/data/llch/Models/OCRFlux-3B/

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
