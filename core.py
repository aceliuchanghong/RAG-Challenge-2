import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from code.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def query_documents(
    question: str,
    table_name: str,
    complicated_question: bool,
    tags: Tuple[str],
    root_path: Path,
) -> Optional[List[Dict[str, Any]]]:
    """
    检索与问题相关的文档的核心逻辑。

    Args:
        question: 用户提出的问题。
        table_name: 要查询的表名。
        complicated_question: 是否为复杂问题。
        tags: 用于过滤的标签元组。
        root_path: 项目的根路径。

    Returns:
        一个包含相关文档字典的列表，如果找不到则返回 None。

    Raises:
        Exception: 当检索过程中发生任何错误时。
    """
    try:
        start_time = time.time()
        pipeline = Pipeline(root_path)

        question_related_docs = pipeline.find_question_related_docs(
            question, table_name, complicated_question, tags
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"检索文档耗时: {elapsed_time:.2f}秒")

        if not question_related_docs or not question_related_docs.get("results"):
            logging.warning(f"未找到与问题 '{question}' 相关的文档。")
            return None

        # 清理返回数据，移除不需要的字段
        results = question_related_docs["results"]
        for item in results:
            if "relevance_score" in item:
                del item["relevance_score"]

        return results

    except Exception as e:
        logging.error(f"在 query_documents 中发生错误: {e}", exc_info=True)
        raise
