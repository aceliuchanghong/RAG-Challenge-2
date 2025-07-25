import httpx
import aiofiles
import asyncio
import os
import fitz
from typing import List, Dict, Union
from termcolor import colored

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
def _convert_pdf_to_images(pdf_path: str) -> List[str]:
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
    output_dir = f"no_git_oic/pdf_images/pdf_images_{base_name}"

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


async def main():
    func_path = os.path.dirname(os.path.abspath(__file__))

    pdf_path = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/NPD2308工艺文件.pdf"

    pdf_img_list = _convert_pdf_to_images(pdf_path)

    results = []
    for pdf_img in pdf_img_list:
        try:
            result = await trans2md_async(
                os.path.join(func_path, pdf_img)
            )  # 2,10,11,31
            print(result)
            results.append(result)
        except Exception as e:
            pass

    save_md_path = os.path.join(func_path, "no_git_oic/save_md/NPD2308.md")

    os.makedirs(os.path.dirname(save_md_path), exist_ok=True)
    with open(save_md_path, "w", encoding="utf-8") as f:
        for result in results:
            if result:
                if "output_files" in result and result["output_files"]:
                    for file_info in result["output_files"]:
                        if "content" in file_info:
                            f.write(file_info["content"] + "\n\n")
    print(f"结果已成功保存到: {save_md_path}")


if __name__ == "__main__":
    """
    cd deal_minerU
    source .venv/bin/activate
    uv run save_md.py
    """
    asyncio.run(main())
