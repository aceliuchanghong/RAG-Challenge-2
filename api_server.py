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
    ser_tab: bool = Field(default=False, description="是否序列化表格")


@app.post("/deal-files/", response_model=Dict[str, Any], summary="一条龙处理文件")
async def deal_files_api(request: DealFilesRequest):
    """
    执行完整的文件处理流水线：
    1.  **读取文件/文件夹**: 将支持的文档格式转换为 Markdown。
    2.  **文本分块**: 将 Markdown 文本分割成带有重叠的、大小固定的块。
    3.  **存入向量库**: 将处理后的数据块嵌入并保存到指定的 LanceDB 表中。
    """
    try:
        result = await process_files_in_pipeline(
            input_path=request.input_path,
            table_name=request.table_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            tags=request.tags,
            ser_tab=request.ser_tab,
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

"""
