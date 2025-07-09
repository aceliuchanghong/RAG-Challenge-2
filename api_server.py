from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


from core import query_documents, process_files_in_pipeline


app = FastAPI(
    title="RAG API",
    description="一个用于文档 rag 处理的 API。",
    version="1.0.0",
)

root_path = Path.cwd()

# from transformers import AutoModelForImageTextToText, AutoProcessor
# from PIL import Image
# from io import BytesIO
# import torch
# import time
# try:
#     print("正在加载模型，请稍候...")
#     model_dir = "/mnt/data/llch/Models/Nanonets-OCR-s"

#     model = AutoModelForImageTextToText.from_pretrained(
#         model_dir,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         # attn_implementation="flash_attention_2",
#     )
#     model.eval()
#     processor = AutoProcessor.from_pretrained(model_dir)
#     print("加载模型成功...")

# except Exception as e:
#     print(f"模型加载失败: {e}")
#     model = None
#     processor = None


# def ocr_image(image: Image.Image, prompt: str, max_new_tokens: int = 4096):
#     """
#     使用加载好的模型对单个 PIL.Image 对象进行 OCR 处理。
#     """
#     if not all([model, processor]):
#         raise RuntimeError("模型未能成功加载，无法处理请求。")

#     # 构建符合模型规范的消息格式
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": prompt},
#             ],
#         },
#     ]
#     print(f"{messages}")

#     # 预处理文本和图像
#     start_time = time.time()
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     # 使用模型生成内容
#     output_ids = model.generate(
#         **inputs, max_new_tokens=max_new_tokens, do_sample=False
#     )

#     # 从输出中移除输入部分
#     input_len = inputs["input_ids"].shape[1]
#     generated_ids = output_ids[:, input_len:]

#     # 解码生成结果
#     output_text = processor.batch_decode(
#         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"图片识别耗时: {elapsed_time:.2f}秒")

#     return output_text[0]


# @app.post("/ocr/", summary="图像文字识别")
# async def perform_ocr(
#     file: UploadFile = File(..., description="需要进行 OCR 的图片文件 (JPG, PNG等)"),
#     prompt: str = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
# ):
#     """
#     上传一张图片，接口将返回识别出的 Markdown 格式文本。
#     """
#     # 验证上传的是否是图片
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="上传的文件不是有效的图片格式。")

#     try:
#         # 读取上传的文件内容并转换为 PIL Image 对象
#         contents = await file.read()
#         image = Image.open(BytesIO(contents)).convert("RGB")

#         # 调用 OCR 函数处理图片
#         result_text = ocr_image(image, prompt)

#         return {"filename": file.filename, "content": result_text}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"处理图片时发生错误: {str(e)}")


class DealFilesRequest(BaseModel):
    """/deal-files/ 接口的请求模型"""

    input_path: str = Field(
        ..., description="服务器上待处理的文件或文件夹的绝对或相对路径。"
    )
    table_name: str = Field(
        default="file_chunks", description="数据要存入的 LanceDB 表名。"
    )
    chunk_size: int = Field(default=300, description="文本分块的大小。")
    chunk_overlap: int = Field(default=50, description="文本分块的重叠大小。")
    tags: Optional[List[str]] = Field(
        default_factory=list, description="要附加到文档块的标签列表。"
    )


@app.post("/deal-files/", response_model=Dict[str, Any], summary="一条龙处理文件")
async def deal_files_api(request: DealFilesRequest):
    """
    执行完整的文件处理流水线：
    1.  **读取文件/文件夹**: 将支持的文档格式转换为 Markdown。
    2.  **文本分块**: 将 Markdown 文本分割成带有重叠的、大小固定的块。
    3.  **存入向量库**: 将处理后的数据块嵌入并保存到指定的 LanceDB 表中。
    """
    try:
        result = process_files_in_pipeline(
            input_path=request.input_path,
            table_name=request.table_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            tags=request.tags,
            root_path=root_path,
        )
        return result
    except ValueError as e:
        # 捕获由无效输入（如路径不存在）引起的特定错误
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获所有其他在处理过程中可能发生的错误
        raise HTTPException(status_code=500, detail=f"文件处理流程中发生严重错误: {e}")


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户提出的问题")
    table_name: str = Field(default="file_chunks", description="要查询的数据库表名")
    complicated_question: bool = Field(
        default=False, description="是否作为复杂问题处理"
    )
    tags: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="用于过滤文档的标签列表",
        examples=[("python", "api")],
    )


@app.post("/get-docs/", response_model=Optional[List[Dict[str, Any]]])
async def get_docs_api(request: QueryRequest):
    """
    根据输入的问题和标签，从指定的表中检索相关文档。
    """
    try:
        documents = query_documents(
            question=request.question,
            table_name=request.table_name,
            complicated_question=request.complicated_question,
            tags=request.tags,
            root_path=root_path,
        )

        if documents is None:
            return []

        return documents

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部发生错误: {e}")


@app.get("/")
def read_root():
    return {"message": "欢迎使用RAG API 请访问 /docs 查看 API 文档。"}


"""
uvicorn api_server:app --host 0.0.0.0 --port 5000
nohup uvicorn api_server:app --host 0.0.0.0 --port 5000 > no_git_oic/api_server.log 2>&1 &
ps -ef | grep api_server
lsof -i :5000

uv run z_utils/remove_comments.py \
    --input api_server.py \
    --output 00.py
        
# 进行文档检索
curl -X 'POST' \
  'http://127.0.0.1:5000/get-docs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "CaTiO3解释?",
    "table_name": "mp_database",
    "complicated_question": false,
    "tags": []
  }'

# 处理服务器上文件夹
curl -X 'POST' \
  'http://127.0.0.1:5000/deal-files/' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "no_git_oic/test_files/",
    "table_name": "my_new_knowledge_base",
    "tags": ["project_alpha", "q2_report"]
  }'

# 处理单个文件
curl -X 'POST' \
  'http://127.0.0.1:5000/deal-files/' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "no_git_oic/test_files/流式细胞制备方案.pdf",
    "table_name": "my_new_knowledge_base"
  }'

curl -X POST \
    -F "file=@/mnt/data/llch/my_lm_log/no_git_oic/images/C23206C4_page_1.png" \
    http://127.0.0.1:5000/ocr/
"""
