import os
import json
from pathlib import Path
from tqdm import tqdm
import lancedb
from datetime import datetime
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

    # 核心文本内容
    text: Optional[str]  # 原始文本块内容
    text_for_fts: Optional[str]  # 分词后的文本，用于FTS索引

    # 向量表示
    vector: Vector(2560)  # 文本对应的向量

    # 元数据标识
    report_sha1: Optional[str]  # 该文本块所属报告的SHA1标识符
    chunk_id: Optional[int]  # 文本块在原报告中的索引位置

    # 标签与分类
    tags: Optional[List[str]]  # 例如: ['2024', '年报', '财务', '公司A']

    # 备用字段
    reserved_field_1: Optional[str]  # 备用字段1
    reserved_field_2: Optional[str]  # 备用字段2

    # 时间与状态管理
    last_modified_date: Optional[datetime]  # 文档创建时间，用于时间过滤
    is_deleted: Optional[bool] = False  # 是否被删除的标志位


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
        self.table = None

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

    def _open_or_create_table(self, table_name: str = "file_chunks"):
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
            all_report_paths = list(path.glob("*.jsonl"))
        elif path.is_file() and path.suffix == ".jsonl":
            all_report_paths = [path]
        else:
            raise ValueError(
                f"Invalid input: {report_or_reports_dir} 是无效的文件或目录，且不为 .jsonl 文件。"
            )

        all_data_to_add = []

        for report_path in tqdm(all_report_paths, desc="[1/3] 解析报告并生成向量"):
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            sha1_name = report_data.get("file_hash")
            chunks = report_data.get("chunks", [])
            if not sha1_name or not chunks:
                print(
                    f"警告: 报告 {report_path} 缺少 'file_hash' 或 'chunks' 字段。跳过此报告。"
                )
                continue

            # 提取文本内容用于生成向量
            text_chunks = [chunk["content"] for chunk in chunks]

            # 如果json文件中没有 'tags' 键，则默认为一个空列表
            report_tags = report_data.get("tags", [])
            # print(f"{report_tags}")

            embeddings = self._get_embeddings(text_chunks)

            if len(chunks) != len(embeddings):
                print(f"警告: 报告 {sha1_name} 的文本块和向量数量不匹配。跳过此报告。")
                continue

            # 直接遍历 chunks 和 embeddings，使代码更清晰
            for chunk, vector in zip(chunks, embeddings):
                text = chunk.get("content")
                if not text:
                    continue  # 如果块内容为空则跳过

                segmented_text = " ".join(jieba.cut_for_search(text))

                data_dict = {
                    "text": text,
                    "text_for_fts": segmented_text,
                    "vector": vector,
                    "report_sha1": sha1_name,
                    "chunk_id": chunk.get("page_num"),
                    "tags": report_tags,  # 从报告级别继承的tags
                    "last_modified_date": datetime.now(),  # 设置当前时间
                    "is_deleted": False,  # 数据默认为未删除
                    "reserved_field_1": None,  # 备用字段默认为 None
                    "reserved_field_2": None,  # 备用字段默认为 None
                }
                all_data_to_add.append(data_dict)
                # ===============================================================

        if all_data_to_add:
            print(
                f"[2/3] 正在向 LanceDB 表中合并 {len(all_data_to_add)} 个数据块 (Upsert)..."
            )
            # 使用 merge_insert 可以保证数据的唯一性（基于report_sha1和chunk_id）
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
        根据目标删除 LanceDB 中的数据 (物理硬删除)。
        注意：此方法执行的是物理删除，而不是将 is_deleted 标志设为 True。
        """
        if table_name not in self.db.table_names():
            print(f"Table '{table_name}' not found. Nothing to delete.")
            return

        self.table = self.db.open_table(table_name)
        sha1s_to_delete = []

        if isinstance(target, list):
            sha1s_to_delete = target
        elif isinstance(target, str) and not Path(target).exists():
            sha1s_to_delete = [target]
        else:
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

    def soft_delete_reports(
        self, target: Union[str, Path, List[str]], table_name="file_chunks"
    ):
        """
        根据目标对 LanceDB 中的数据进行软删除 (将 is_deleted 标志设为 True)。

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
        sha1s_to_update = []

        # --- 这部分逻辑与硬删除一致，用于解析目标并获取SHA1列表 ---
        if isinstance(target, list):
            sha1s_to_update = target
        elif isinstance(target, str) and not Path(target).exists():
            sha1s_to_update = [target]
        else:
            path = Path(target)
            if path.is_file() and path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    sha1s_to_update.append(json.load(f)["file_hash"])
            elif path.is_dir():
                for report_path in path.glob("*.jsonl"):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            sha1s_to_update.append(json.load(f)["file_hash"])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not read SHA1 from {report_path}: {e}")
            else:
                raise ValueError(f"Invalid target for deletion: {target}")

        if not sha1s_to_update:
            print("No valid SHA1s found for soft deletion.")
            return

        # --- 核心修改：从 delete() 改为 update() ---
        formatted_sha1s = ", ".join([f"'{s}'" for s in sha1s_to_update])
        update_condition = f"report_sha1 IN ({formatted_sha1s})"

        print(
            f"正在对表 '{table_name}' 中的 {len(sha1s_to_update)} 个报告进行软删除..."
        )
        print(f"执行更新条件: {update_condition}")

        try:
            # 将 is_deleted 字段更新为 True
            self.table.update(where=update_condition, values={"is_deleted": True})
            print("软删除操作完成。")
        except Exception as e:
            print(f"An error occurred during soft deletion: {e}")

    def soft_delete_chunk(
        self, report_sha1: str, chunk_id: int, table_name="file_chunks"
    ):
        """
        根据报告SHA1和块ID软删除单个数据块。
        """
        if table_name not in self.db.table_names():
            print(f"Table '{table_name}' not found. Nothing to delete.")
            return

        self.table = self.db.open_table(table_name)
        update_condition = f"report_sha1 = '{report_sha1}' AND chunk_id = {chunk_id}"

        print(f"正在从表 '{table_name}' 中软删除块...")
        print(f"执行更新条件: {update_condition}")

        try:
            self.table.update(where=update_condition, values={"is_deleted": True})
            print("软删除操作完成。")
        except Exception as e:
            print(f"An error occurred during soft deletion: {e}")

    def keyword_search(
        self,
        query: str,
        limit: int = 8,
        tags_filter: Optional[List[str]] = [],
        do_print: bool = True,
    ) -> List[LanceDBSchema]:
        """
        执行基于关键字的全文搜索 (FTS)。
        - 默认过滤掉被软删除的记录。
        - 可根据 tags 进行额外过滤。
        """
        if do_print:
            print(f"--- 关键字搜索: '{query}' ---")
            if tags_filter:
                print(f"--- Tags 过滤: {tags_filter} ---")

        conditions = ["is_deleted = false"]  # 默认条件：只搜索未被删除的
        if tags_filter:
            # 为每个 tag 创建一个独立的 array_contains 条件
            for tag in tags_filter:
                conditions.append(f"array_contains(tags, '{tag}')")

        final_where_clause = " AND ".join(conditions)
        if do_print:
            print(f"--- 生成的 WHERE 子句: {final_where_clause} ---")

        segmented_query = " ".join(jieba.cut_for_search(query))
        # print(f"{final_where_clause},{segmented_query}")

        results = (
            self.table.search(segmented_query)
            .where(final_where_clause)  # 应用所有过滤条件
            .limit(limit)
            .to_pydantic(LanceDBSchema)
        )

        if do_print:
            if not results:
                print("未找到匹配的结果。\n")
            for res in results:
                print(
                    f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}, Tags: {res.tags}]"
                )
                print(f"    文本: {res.text[:150]}...\n")
        return results

    def vector_search(
        self,
        query: str,
        limit: int = 8,
        tags_filter: Optional[List[str]] = None,
        do_print: bool = True,
    ) -> List[LanceDBSchema]:
        """
        执行基于向量的语义相似度搜索。
        - 默认过滤掉被软删除的记录。
        - 可根据 tags 进行额外过滤。
        """
        if do_print:
            print(f"--- 向量搜索: '{query}' ---")
            if tags_filter:
                print(f"--- Tags 过滤: {tags_filter} ---")

        conditions = ["is_deleted = false"]
        if tags_filter:
            for tag in tags_filter:
                conditions.append(f"array_contains(tags, '{tag}')")
        final_where_clause = " AND ".join(conditions)

        if do_print:
            print(f"--- 生成的 WHERE 子句: {final_where_clause} ---")

        query_vector = self._get_embeddings([query])[0]

        results = (
            self.table.search(query_vector)
            .where(final_where_clause)  # 应用所有过滤条件
            .limit(limit)
            .to_pydantic(LanceDBSchema)
        )

        if do_print:
            if not results:
                print("未找到匹配的结果。\n")
            for res in results:
                print(
                    f"  - [报告SHA1: {res.report_sha1}, 块ID: {res.chunk_id}, Tags: {res.tags}]"
                )
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
    # ingestor.process_and_ingest_reports(REPORTS_DIR)
    print("=" * 50)

    # 连接数据库
    ingestor._open_or_create_table()

    query = "流式细胞制备如何操作?"
    do_print = True
    tags_to_find = []
    tags_to_find = ["test1", "test2"]
    tags_to_find = ["test1"]

    # a) 关键字搜索 (类似于 BM25)
    results_fts = ingestor.keyword_search(
        query, tags_filter=tags_to_find, do_print=do_print
    )

    # b) 向量搜索 (语义搜索)
    results_vec = ingestor.vector_search(
        query, tags_filter=tags_to_find, do_print=do_print
    )
