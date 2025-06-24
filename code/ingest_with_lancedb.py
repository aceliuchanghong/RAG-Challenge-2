import os
import json
from pathlib import Path
from tqdm import tqdm
import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt
from typing import List, Union, Optional
import jieba


load_dotenv()

LANCEDB_PATH = Path("./lancedb")
TEMP_DIR = LANCEDB_PATH.parent / "lancedb_tmp"
TEMP_DIR.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(TEMP_DIR)
"""
lancedb:https://lancedb.github.io/lancedb/
https://lancedb.github.io/lancedb/python/python/
"""


class LanceDBSchema(LanceModel):
    """定义数据模型 (Schema)"""

    text: Optional[str]  # 原始文本块内容
    text_for_fts: Optional[str]  # 分词后的文本，用于FTS索引
    vector: Vector(2560)  # 文本对应的向量
    report_sha1: Optional[str]  # 该文本块所属报告的SHA1标识符
    chunk_id: Optional[int]  # 文本块在原报告中的索引位置


class LanceDBIngestor:
    """
    LanceDB 数据构建与存储器

    一个统一的处理器，负责：
    1. 从JSON报告中读取文本块。
    2. 使用EMB API为文本块生成向量。
    3. 将文本、向量和元数据存入一个单一的 LanceDB 表中。
    4. 在文本字段上创建全文搜索 FTS 索引。
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "./lancedb",
    ):
        self.emb = OpenAI(
            api_key=os.getenv("EMB_API_KEY"),
            base_url=os.getenv("EMB_BASE_URL"),
            max_retries=2,
        )
        self.db = lancedb.connect(db_path)
        self.table_name = None

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(2))
    def _get_embeddings(
        self, texts: List[str], model: str = "Qwen3-Embedding-4B"
    ) -> List[List[float]]:
        """
        为文本列表批量获取嵌入向量。
        """
        if not texts:
            return []

        # 过滤掉可能存在的空字符串
        texts = [t.replace("\n", " ") for t in texts if t.strip()]
        if not texts:
            return []

        response = self.emb.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]

    def _open_or_create_table(self, table_name: str):
        """内部辅助函数，用于打开或创建表"""
        try:
            if table_name in self.db.table_names():
                self.table = self.db.open_table(table_name)
            else:
                print(f"Table '{table_name}' not found, creating new one.")
                self.table = self.db.create_table(table_name, schema=LanceDBSchema)
        except lancedb.errors.LanceDBError as e:
            print(f"Error opening or creating table '{table_name}': {e}")
            raise

    def process_and_ingest_reports(
        self, report_or_reports_dir: Union[str, Path], table_name="file_chunks"
    ):
        """
        处理所有报告，并将数据分批次存入 LanceDB。
        """
        self._open_or_create_table(table_name)
        path = Path(report_or_reports_dir)

        if path.is_dir():
            # 如果是目录，递归查找所有 .jsonl 文件
            all_report_paths = list(path.glob("*.jsonl"))
        elif path.is_file() and path.suffix == ".jsonl":
            # 如果是单个 .jsonl 文件
            all_report_paths = [path]
        else:
            raise ValueError(
                f"Invalid input: {report_or_reports_dir} 是无效的文件或目录，且不为 .jsonl 文件。"
            )

        # 所有待添加的数据
        all_data_to_add = []

        for report_path in tqdm(all_report_paths, desc="[1/3] 解析报告并生成向量"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            sha1_name = report_data["file_hash"]
            text_chunks = [chunk["content"] for chunk in report_data["chunks"]]
            page_nums = [chunk["page_num"] for chunk in report_data["chunks"]]

            # 为当前报告的所有文本块批量获取向量
            embeddings = self._get_embeddings(text_chunks)

            # 确保向量数量和文本块数量一致
            if len(text_chunks) != len(embeddings):
                print(f"警告: 报告 {sha1_name} 的文本块和向量数量不匹配。跳过此报告。")
                continue

            # 准备要插入的数据
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
            print(
                f"[2/3] 正在向 LanceDB 表中合并 {len(all_data_to_add)} 个数据块 (Upsert)..."
            )
            self.table.merge_insert(
                on=["report_sha1", "chunk_id"]
            ).when_matched_update_all().when_not_matched_insert_all().execute(
                all_data_to_add
            )

        print("[3/3] 正在创建全文搜索 (FTS) 索引...")
        self.table.create_fts_index("text_for_fts", replace=True)
        print("FTS 索引创建完成。")

        print(f"处理了 {len(all_report_paths)} 个报告，数据库构建完成！\n")

    def delete_reports(
        self, target: Union[str, Path, List[str]], table_name="file_chunks"
    ):
        """
        根据目标删除 LanceDB 中的数据。

        目标可以是:
        - 单个报告的 SHA1 (str)
        - 多个报告 SHA1 的列表 (List[str])
        - 单个 .jsonl 文件的路径 (str or Path)
        - 包含多个 .jsonl 文件的目录路径 (str or Path)
        """
        if table_name not in self.db.table_names():
            print(f"Table '{table_name}' not found. Nothing to delete.")
            return

        self.table = self.db.open_table(table_name)
        sha1s_to_delete = []

        if isinstance(target, list):
            sha1s_to_delete = target
        elif isinstance(target, str) and not Path(target).exists():
            # 假定这是一个单独的SHA1字符串，而不是一个路径
            sha1s_to_delete = [target]
        else:
            # 处理文件或目录路径
            path = Path(target)
            if path.is_file() and path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    sha1s_to_delete.append(json.load(f)["file_hash"])
            elif path.is_dir():
                for report_path in path.glob("*.jsonl"):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            sha1s_to_delete.append(json.load(f)["file_hash"])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not read SHA1 from {report_path}: {e}")
            else:
                raise ValueError(f"Invalid target for deletion: {target}")

        if not sha1s_to_delete:
            print("No valid SHA1s found for deletion.")
            return

        # 构建 SQL WHERE IN 子句
        # 例如: "report_sha1 IN ('sha1_A', 'sha1_B')"
        formatted_sha1s = ", ".join([f"'{s}'" for s in sha1s_to_delete])
        delete_condition = f"report_sha1 IN ({formatted_sha1s})"

        print(f"正在从表 '{table_name}' 中删除 {len(sha1s_to_delete)} 个报告的数据...")
        print(f"执行删除条件: {delete_condition}")

        try:
            self.table.delete(delete_condition)
            print("删除操作完成。")
        except Exception as e:
            print(f"An error occurred during deletion: {e}")

    def delete_chunk(self, report_sha1: str, chunk_id: int, table_name="file_chunks"):
        """
        根据报告SHA1和块ID删除单个数据块。
        """
        if table_name not in self.db.table_names():
            print(f"Table '{table_name}' not found. Nothing to delete.")
            return

        self.table = self.db.open_table(table_name)

        delete_condition = f"report_sha1 = '{report_sha1}' AND chunk_id = {chunk_id}"

        print(f"正在从表 '{table_name}' 中删除块...")
        print(f"执行删除条件: {delete_condition}")

        try:
            self.table.delete(delete_condition)
            print("删除操作完成。")
        except Exception as e:
            print(f"An error occurred during deletion: {e}")

    def keyword_search(
        self, query: str, limit: int = 2, do_print: bool = True
    ) -> List[LanceModel]:
        """
        执行基于关键字的全文搜索 (FTS)。
        """
        if do_print:
            print(f"--- 关键字搜索: '{query}' ---")
        segmented_query = " ".join(jieba.cut_for_search(query))
        results = (
            self.table.search(segmented_query).limit(limit).to_pydantic(LanceDBSchema)
        )
        if do_print:
            for res in results:
                print(f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}]")
                print(f"    文本: {res.text[:150]}...\n")
        return results

    def vector_search(
        self, query: str, limit: int = 1, do_print: bool = True
    ) -> List[LanceModel]:
        """
        执行基于向量的语义相似度搜索。
        """
        if do_print:
            print(f"--- 向量搜索: '{query}' ---")
        query_vector = self._get_embeddings([query])[0]
        results = (
            self.table.search(query_vector).limit(limit).to_pydantic(LanceDBSchema)
        )
        if do_print:
            for res in results:
                print(f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}]")
                print(f"    文本: {res.text[:150]}...\n")
        return results


if __name__ == "__main__":
    """
    uv run code/ingest_with_lancedb.py
    """
    REPORTS_DIR = Path("output/chunked_md")

    os.makedirs(LANCEDB_PATH, exist_ok=True)

    print("=" * 50)
    print("开始构建 LanceDB 数据库...")
    ingestor = LanceDBIngestor()
    ingestor.process_and_ingest_reports(REPORTS_DIR)
    print("=" * 50)

    # 连接数据库
    db_for_query = lancedb.connect(LANCEDB_PATH)
    table_for_query = db_for_query.open_table("file_chunks")

    # a) 关键字搜索 (类似于 BM25)
    keyword_query = "锥形离心试管"
    results_fts = ingestor.keyword_search(keyword_query)

    # b) 向量搜索 (语义搜索)
    vector_query = "锥形离心试管"
    results_vec = ingestor.vector_search(keyword_query)
