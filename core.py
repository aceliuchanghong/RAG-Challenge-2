import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import json

from code.pipeline import Pipeline
from code.common import DocumentType, extract_hash
from z_utils.hash_x import compute_mdhash_id

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


async def process_files_in_pipeline(
    input_path: str,
    table_name: str = "file_chunks",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    tags: List[str] = [],
    ser_tab: bool = False,
    root_path: Path = "./",
) -> Dict[str, Any]:
    """
    一条龙处理文件：读取 -> 分块 -> 保存到 LanceDB。
    此版本精确集成了提供的 _process_file 和 _process_and_chunk_file 的逻辑。

    Args:
        input_path (str): 待处理的文件或文件夹路径。
        table_name (str): 要存入的目标 LanceDB 表名。
        chunk_size (int): 文本分块大小。
        chunk_overlap (int): 文本分块重叠大小。
        tags (List[str]): 要附加到每个数据块的标签。
        root_path (Path): 项目根路径。

    Returns:
        Dict[str, Any]: 包含处理结果摘要的字典。
    """
    start_time = time.time()
    logging.info(
        f"开始一条龙文件处理流程，输入路径: '{input_path}', 表名: '{table_name}'"
    )

    pipeline = Pipeline(root_path)
    source_path = Path(input_path)

    temp_output_dir = (
        root_path / "no_git_oic/api_deal_files" / f"{compute_mdhash_id(input_path)}"
    )
    md_output_dir = temp_output_dir / "md"
    chunked_output_dir = temp_output_dir / "chunked_md"
    source_backup_dir = temp_output_dir / "original"

    for dir_path in [md_output_dir, chunked_output_dir, source_backup_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # === 步骤 1: 读取文件并转换为 Markdown (集成 _process_file 逻辑) ===
        logging.info("步骤 1/3: 读取文件并转换为 Markdown...")
        if not source_path.exists():
            raise ValueError(f"输入路径不存在: '{input_path}'")

        allowed_suffixes = {
            f".{ext}" for doc_type in DocumentType for ext in doc_type.value
        }
        files_to_process = []
        if source_path.is_file():
            if source_path.suffix in allowed_suffixes:
                files_to_process.append(source_path)
        elif source_path.is_dir():
            files_to_process.extend(
                [
                    item
                    for item in source_path.iterdir()
                    if item.is_file() and item.suffix in allowed_suffixes
                ]
            )

        if not files_to_process:
            logging.warning(f"在 '{input_path}' 中未找到可处理的文件。")
            return {
                "status": "completed",
                "message": "No processable files found.",
                "processed_files_count": 0,
            }

        md_file_paths = []
        for file_path in files_to_process:
            logging.info(f"正在处理文件: {file_path}")
            content = pipeline.read_file(file_path)  # 使用 pipeline.read_file 获取内容
            if content:
                mdhash_id = compute_mdhash_id(content)  # 从内容计算哈希
                md_file_path = md_output_dir / f"md_{mdhash_id}.md"
                md_file_path.write_text(content, encoding="utf-8")
                md_file_paths.append(md_file_path)

                # 备份原始文件
                dest_file = source_backup_dir / f"{mdhash_id}{file_path.suffix}"
                shutil.copy(file_path, dest_file)
                logging.info(f"成功处理并保存内容到 {md_file_path}")

        logging.info(f"成功转换 {len(md_file_paths)} 个文件到 Markdown 格式。")

        # === 步骤 2: 对 Markdown 文件进行分块 (集成 _process_and_chunk_file 逻辑) ===
        logging.info("步骤 2/3: 对 Markdown 文件进行分块...")
        total_chunks = 0
        if md_file_paths:
            for md_path in md_file_paths:
                logging.info(f"正在分块 Markdown 文件: {md_path.name}")
                # 使用 pipeline.chunk_md_file
                chunks = await pipeline.chunk_md_file(
                    str(md_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tags=tuple(tags),  # 确保是元组
                    ser_tab=ser_tab,
                )

                if chunks:
                    file_hash = extract_hash(md_path.name)
                    jsonl_output_path = (
                        chunked_output_dir / f"chunked_{file_hash}.jsonl"
                    )

                    # 精确复制逻辑: 将整个列表转为JSON字符串，写入单行
                    with jsonl_output_path.open("w", encoding="utf-8") as f:
                        json_line = json.dumps(chunks, ensure_ascii=False)
                        f.write(json_line + "\n")

                    total_chunks += len(chunks)
                    logging.info(f"成功分块文件并保存到 {jsonl_output_path}")

        logging.info(
            f"成功将 {len(md_file_paths)} 个 Markdown 文件分块，总计 {total_chunks} 个数据块。"
        )

        # === 步骤 3: 保存 .jsonl 文件到 LanceDB ===
        logging.info(f"步骤 3/3: 保存数据块到 LanceDB 表 '{table_name}'...")
        if total_chunks > 0:
            pipeline.save2lacncedb(
                report_or_reports_dir=str(chunked_output_dir), table_name=table_name
            )
            logging.info(f"数据成功保存到表 '{table_name}'。")
        else:
            logging.warning("没有生成任何数据块，无需保存到数据库。")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"一条龙处理流程完成，总耗时: {elapsed_time:.2f} 秒。")

        return {
            "status": "success",
            "message": "Files processed and saved to database successfully.",
            "input_path": input_path,
            "table_name": table_name,
            "processed_files_count": len(md_file_paths),
            "total_chunks_created": total_chunks,
            "duration_seconds": round(elapsed_time, 2),
        }

    except Exception as e:
        logging.error(
            f"在 'process_files_in_pipeline' 中发生严重错误: {e}", exc_info=True
        )
        raise
