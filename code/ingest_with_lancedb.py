import os
import json
from typing import List
from pathlib import Path
from tqdm import tqdm
import lancedb
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt
from typing import List, Union
import jieba

from dotenv import load_dotenv

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

    text: str  # 原始文本块内容
    text_for_fts: str  # 分词后的文本，用于FTS索引
    vector: Vector(2560)  # 文本对应的向量
    report_sha1: str  # 该文本块所属报告的SHA1标识符
    chunk_id: int  # 文本块在原报告中的索引位置


class LanceDBIngestor:
    """
    LanceDB 数据构建与存储器

    一个统一的处理器，负责：
    1. 从JSON报告中读取文本块。
    2. 使用EMB API为文本块生成向量。
    3. 将文本、向量和元数据存入一个单一的 LanceDB 表中。
    4. 在文本字段上创建全文搜索 FTS 索引。
    """

    def __init__(self, db_path: Union[str, Path] = "./lancedb"):
        # 设置 OpenAI 客户端
        load_dotenv()
        self.llm = OpenAI(
            api_key=os.getenv("EMB_API_KEY"),
            base_url=os.getenv("EMB_BASE_URL"),
            max_retries=2,
        )
        # 连接到 LanceDB 数据库（如果不存在，则会自动创建）
        self.db = lancedb.connect(db_path)
        self.table = self.db.create_table(
            "file_chunks", schema=LanceDBSchema, mode="overwrite"
        )

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
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

        response = self.llm.embeddings.create(input=texts, model=model)
        return [embedding.embedding for embedding in response.data]

    def process_and_ingest_reports(self, all_reports_dir: Path):
        """
        处理所有报告，并将数据分批次存入 LanceDB。
        """
        all_report_paths = list(all_reports_dir.glob("*.jsonl"))
        print(f"{all_report_paths}")

        # 使用一个列表来收集所有待添加的数据
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

        # 将所有数据一次性添加到表中，效率最高
        if all_data_to_add:
            print(f"\n[2/3] 正在向 LanceDB 表中添加 {len(all_data_to_add)} 个数据块...")
            self.table.add(all_data_to_add)
            print("数据添加完成。")

        # 在 'text_for_fts' 字段上创建全文搜索 (FTS) 索引，用于关键字搜索
        print("[3/3] 正在创建全文搜索 (FTS) 索引...")
        self.table.create_fts_index("text_for_fts", replace=True)
        print("FTS 索引创建完成。")

        print(f"\n处理了 {len(all_report_paths)} 个报告，数据库构建完成！")

    def keyword_search(self, query: str, limit: int = 2):
        """
        执行基于关键字的全文搜索 (FTS)。
        """
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
        """
        执行基于向量的语义相似度搜索。
        """
        print(f"\n--- 向量搜索: '{query}' ---")
        # 1. 获取查询语句的向量
        query_vector = self._get_embeddings([query])[0]
        # 2. 在向量列上进行搜索
        results = (
            self.table.search(query_vector).limit(limit).to_pydantic(LanceDBSchema)
        )
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
    ingestor = LanceDBIngestor(db_path=LANCEDB_PATH)
    ingestor.process_and_ingest_reports(REPORTS_DIR)
    print("=" * 50)

    #  ingestion 和 searching 可以是两个独立的脚本
    print("\n数据库已就绪，开始演示搜索功能...")

    # 重新连接数据库进行查询（模拟一个独立的应用）
    db_for_query = lancedb.connect(LANCEDB_PATH)
    table_for_query = db_for_query.open_table("file_chunks")

    # a) 关键字搜索 (类似于 BM25)
    keyword_query = "锥形离心试管"
    results_fts = ingestor.keyword_search(keyword_query)
    print(results_fts)

    # b) 向量搜索 (语义搜索)
    vector_query = "锥形离心试管"
    results_vec = ingestor.vector_search(keyword_query)
    print(results_vec)
