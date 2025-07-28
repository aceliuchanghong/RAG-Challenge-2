import httpx
import aiofiles
import asyncio
import os
import fitz
from typing import List, Dict, Union
from termcolor import colored
from pathlib import Path
from typing import List, Union, Any, Dict
import tempfile
import base64
import hashlib

from sqlite_cache import cache_to_sqlite, cache_to_sqlite_async


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
你是一位 MLCC 专家，你的任务是分析图片，生成结构化的JSON数据，以便于后续的检索和问答。

# 输出要求：
请严格按照以下JSON格式输出你的分析结果，不要添加任何额外的解释。

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


@cache_to_sqlite(debug=False)
def _compute_md5_hash(content: str, prefix: str = "") -> str:
    """计算内容的MD5哈希值。"""
    return prefix + hashlib.md5(content.encode()).hexdigest()


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

    tasks = []
    print(f"准备处理的图片: {image_paths_to_process}")
    for img_path in image_paths_to_process:
        tasks.append(trans2md_async(img_path))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 整合结果并保存到Markdown文件
    if not input_identifier:
        # 如果标识符为空（例如，输入一个空列表），则创建一个
        input_identifier = str(asyncio.get_running_loop().time())

    # 使用输入内容的哈希值作为文件名，避免重复
    file_hash_id = _compute_md5_hash(input_identifier)
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


# async def main():
#     """
#     异步程序的主入口
#     """
#     image_to_test = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J2.png"
#     image_to_test = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/image.png"
#     # image_to_test = "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_2.png"
#     # image_to_test = "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_31.png"

#     # 检查示例文件是否存在
#     if not os.path.exists(image_to_test):
#         print(f"示例文件不存在: {image_to_test}")
#         return

#     try:
#         print(f"正在处理图片: {image_to_test}")

#         result = await trans2md_async(image_to_test)

#         print("\n--- API 响应结果 ---")
#         print(result)

#     except FileNotFoundError as e:
#         print(e)
#     except httpx.RequestError as e:
#         print(f"请求失败: {e}")
#     except Exception as e:
#         print(f"发生未知错误: {e}")


# async def main():
#     func_path = os.path.dirname(os.path.abspath(__file__))

#     pdf_path = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/NPD2308工艺文件.pdf"

#     pdf_img_list = _convert_pdf_to_images(pdf_path)

#     results = []
#     for pdf_img in pdf_img_list:
#         try:
#             result = await trans2md_async(
#                 os.path.join(func_path, pdf_img)
#             )  # 2,10,11,31
#             print(result)
#             results.append(result)
#         except Exception as e:
#             pass

#     save_md_path = os.path.join(func_path, "no_git_oic/save_md/NPD2308.md")

#     os.makedirs(os.path.dirname(save_md_path), exist_ok=True)
#     with open(save_md_path, "w", encoding="utf-8") as f:
#         for result in results:
#             if result:
#                 if "output_files" in result and result["output_files"]:
#                     for file_info in result["output_files"]:
#                         if "content" in file_info:
#                             f.write(file_info["content"] + "\n\n")
#     print(f"结果已成功保存到: {save_md_path}")


if __name__ == "__main__":
    """
    cd deal_minerU
    source .venv/bin/activate
    uv run save_md.py
    """
    # asyncio.run(main())

    pdf_path = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/NPD2308工艺文件.pdf"
    image_to_test = "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_2.png"
    image_to_test2 = [
        "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_2.png",
        "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_1.png",
    ]
    asyncio.run(get_std_md(image_to_test2))
