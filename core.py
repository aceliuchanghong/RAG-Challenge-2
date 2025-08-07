import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import shutil
import json

import httpx
import aiofiles
import asyncio
import os
import fitz
import base64


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from code.pipeline import Pipeline
from code.common import DocumentType, extract_hash
from z_utils.hash_x import compute_mdhash_id
from z_utils.sqlite_cache import cache_to_sqlite, cache_to_sqlite_async


@cache_to_sqlite_async(debug=False)
async def trans2md_async(
    image_path: str,
    use_visual_model: bool = True,
    visual_model_prompt: str = None,
) -> dict:
    """
    异步调用 trans2md 接口 上传图片并获取Markdown格式的OCR结果。

    参数:
    image_path (str): 本地图片文件的路径。
    use_visual_model (bool): 是否使用视觉模型。
    visual_model_prompt (str): 视觉模型的提示词。

    返回示例:
        {
            "output_files": [
                {
                    "path": "/mnt/data/llch/new_minerU/no_git_oic/upload_files_md_output/image/vlm",
                    "content": "# 流式细胞制备方案\n\n",
                }
            ]
        }
    """
    url = "http://121.205.3.100:5005/trans2md/"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误：找不到文件 '{image_path}'")
    if use_visual_model and not visual_model_prompt:
        visual_model_prompt = """
分析图片，生成结构化的JSON数据，以便于后续的检索和问答。

# 输出要求：
请严格按照以下JSON格式输出分析结果，不要添加任何额外的解释。

{
  "content_description": "（详细描述工艺流程图的逻辑，先什么后什么，它们之间的关系是怎样的。力求详尽、客观。）",
  "key_elements": [
    "（从图片中识别出的核心物体、区域或概念，以列表形式提供）",
    "（例如：金属化区域、空调外机、特定材料层）"
  ],
}
"""

    form_data = {
        "use_visual_model": use_visual_model,
        "visual_model_prompt": visual_model_prompt,
    }

    async with aiofiles.open(image_path, "rb") as image_file:
        files_payload = {
            "files": (
                os.path.basename(image_path),
                await image_file.read(),
                "image/png",
            )
        }

        async with httpx.AsyncClient(timeout=900.0) as client:
            response = await client.post(
                url,
                headers={"accept": "application/json"},
                data=form_data,
                files=files_payload,
            )

    response.raise_for_status()

    return response.json()


@cache_to_sqlite(debug=False)
def _convert_pdf_to_images(pdf_path: str, output_path=None) -> List[str]:
    """
    将PDF文件的每一页转换为PNG图片。

    参数:
    pdf_path (str): PDF文件的路径。

    返回:
    List[str]: 生成的图片文件路径列表。
    """
    # uv pip install PyMuPDF
    pdf_document = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if output_path is None:
        output_dir = f"no_git_oic/pdf_images/pdf_images_{base_name}"
    else:
        output_dir = output_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    print(f"正在将PDF '{pdf_path}' 转换为图片...")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # 设置更高分辨率以获得更清晰的图片
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    print(f"🖼️ PDF已转换为 {len(image_paths)} 张图片，保存在 '{output_dir}' 目录。")
    return image_paths


def _save_base64_image(b64_string: str, output_folder: Path) -> str:
    """解码Base64字符串并保存为图片。"""
    try:
        # 移除MIME类型头部 (e.g., "data:image/png;base64,")
        header, encoded = b64_string.split(",", 1)
        image_data = base64.b64decode(encoded)

        # 从头部推断文件扩展名
        ext = header.split("/")[1].split(";")[0]
        if not ext:
            ext = "png"  # 默认扩展名

        img_path = output_folder / f"from_base64.{ext}"
        with open(img_path, "wb") as f:
            f.write(image_data)
        return str(img_path)
    except Exception as e:
        raise ValueError(f"无效的Base64图片字符串: {e}")


async def get_std_md(
    input_data: Union[str, List[str], Any],
    output_dir: str = "no_git_oic/save_md/",
    processing_dir: str = "no_git_oic/save_files/",
):
    """
    将多种格式的输入 图片、PDF、Base64、文件上传 转换为Markdown文件。

    Args:
        input_data (Union[str, List[str], Any]): 输入数据，可以是：
            - 单个图片或PDF文件的路径 (str)。
            - 多个图片文件路径的列表 (List[str])。
            - Base64编码的图片字符串 (str)。
            - 兼容FastAPI的UploadFile对象 (Any)。
        output_dir (str, optional): Markdown文件的输出目录。
            默认为 "no_git_oic/save_md/"。

    Returns:
        str: 保存的Markdown文件的完整路径。
    """
    image_paths_to_process = []
    input_identifier = ""  # 用于生成唯一哈希文件名

    # 创建用于处理中间文件的指定目录，如果不存在则创建
    save_files_dir = Path(processing_dir)
    save_files_dir.mkdir(parents=True, exist_ok=True)

    # 1. 判断并处理输入数据
    if isinstance(input_data, list):
        # 输入是图片路径列表
        image_paths_to_process.extend(input_data)
        input_identifier = "".join(sorted(input_data))

    elif isinstance(input_data, str):
        if input_data.startswith("data:image"):
            # 输入是Base64字符串
            img_path = _save_base64_image(input_data, save_files_dir)
            if img_path:
                image_paths_to_process.append(img_path)
            input_identifier = input_data
        elif Path(input_data).is_file():
            input_path = Path(input_data)
            input_identifier = str(input_path.resolve())
            if input_path.suffix.lower() == ".pdf":
                # 输入是PDF文件路径
                pdf_images = _convert_pdf_to_images(input_path, save_files_dir)
                image_paths_to_process.extend(pdf_images)
            else:
                # 输入是单个图片路径
                image_paths_to_process.append(str(input_path))
        else:
            raise FileNotFoundError(f"输入路径不存在或不是一个文件: {input_data}")

    # 检查是否为类文件对象（如FastAPI的UploadFile）
    elif hasattr(input_data, "filename") and hasattr(input_data, "read"):
        filename = input_data.filename
        input_identifier = filename

        # 将上传的文件保存到处理目录
        file_path_in_processing_dir = save_files_dir / filename

        # Check if read is a coroutine function
        if asyncio.iscoroutinefunction(input_data.read):
            content = await input_data.read()
        else:
            content = input_data.read()

        with open(file_path_in_processing_dir, "wb") as f:
            f.write(content)

        if filename.lower().endswith(".pdf"):
            pdf_images = _convert_pdf_to_images(
                file_path_in_processing_dir, save_files_dir
            )
            image_paths_to_process.extend(pdf_images)
        else:
            # 假设是图片
            image_paths_to_process.append(str(file_path_in_processing_dir))
    else:
        raise TypeError(f"不支持的输入类型: {type(input_data)}")

    # 2. 异步处理所有准备好的图片
    if not image_paths_to_process:
        print("没有需要处理的图片。")
        return ""

    # 创建一个Semaphore，并将并发数限制为 4
    semaphore = asyncio.Semaphore(4)

    async def process_with_semaphore(img_path):
        # async with 会自动获取和释放 semaphore
        async with semaphore:
            # print(f"开始处理: {img_path}")
            # 等待 trans2md_async 完成
            result = await trans2md_async(img_path)
            # print(f"完成处理: {img_path}")
            return result

    tasks = []
    print(f"准备处理的图片: {image_paths_to_process}")

    for img_path in image_paths_to_process:
        tasks.append(process_with_semaphore(img_path))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 整合结果并保存到Markdown文件
    if not input_identifier:
        # 如果标识符为空（例如，输入一个空列表），则创建一个
        input_identifier = str(asyncio.get_running_loop().time())

    # 使用输入内容的哈希值作为文件名，避免重复
    file_hash_id = compute_mdhash_id(input_identifier)
    save_md_path = Path(output_dir) / f"{file_hash_id}.md"

    # 确保最终输出目录存在
    save_md_path.parent.mkdir(parents=True, exist_ok=True)

    final_content = []
    for result in results:
        if isinstance(result, Exception):
            print(f"处理某个文件时出错: {result}")
            continue

        if result and "output_files" in result and result["output_files"]:
            for file_info in result["output_files"]:
                if "content" in file_info:
                    final_content.append(file_info["content"])

    with open(save_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(final_content))

    print(f"结果已成功保存到: {save_md_path}")
    return str(save_md_path)


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
    tags: Optional[List[str]] = None,
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
    logging.info(f"开始可读文件一条龙文件，表名: '{table_name}', 文件: '{input_path}'")

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
            # logging.info(f"正在处理文件: {file_path}")
            content = pipeline.read_file(file_path)  # 使用 pipeline.read_file 获取内容
            if content:
                mdhash_id = compute_mdhash_id(content)  # 从内容计算哈希
                md_file_path = md_output_dir / f"md_{mdhash_id}.md"
                md_file_path.write_text(content, encoding="utf-8")
                md_file_paths.append(md_file_path)

                # 备份原始文件
                dest_file = source_backup_dir / f"{mdhash_id}{file_path.suffix}"
                shutil.copy(file_path, dest_file)
                logging.info(f"标准md:{md_file_path}")

        # logging.info(f"成功转换 {len(md_file_paths)} 个文件到 Markdown 格式。")

        # === 步骤 2: 对 Markdown 文件进行分块 (集成 _process_and_chunk_file 逻辑) ===
        logging.info("步骤 2/3: 对 Markdown 文件进行分块...")
        total_chunks = 0
        if md_file_paths:
            for md_path in md_file_paths:
                # logging.info(f"正在分块 Markdown 文件: {md_path.name}")
                # 使用 pipeline.chunk_md_file
                chunks = await pipeline.chunk_md_file(
                    str(md_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tags=tuple(tags) if tags is not None else None,
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
                    # logging.info(f"分块文件:{jsonl_output_path}")

        logging.info(
            f"成功将 {len(md_file_paths)} 个 Markdown 文件分块，总计 {total_chunks} 个数据块。"
        )

        # === 步骤 3: 保存 .jsonl 文件到 LanceDB ===
        logging.info("步骤 3/3: 保存数据块到 LanceDB...")
        if total_chunks > 0:
            pipeline.save2lacncedb(
                report_or_reports_dir=str(chunked_output_dir), table_name=table_name
            )
            # logging.info(f"数据成功保存到表 '{table_name}'。")
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
