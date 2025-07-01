from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from core import query_documents


app = FastAPI(
    title="RAG API",
    description="一个用于文档 rag 处理的 API。",
    version="1.0.0",
)

root_path = Path.cwd()


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
uvicorn api_server:app --reload --host 0.0.0.0 --port 5000
nohup uvicorn api_server:app --reload --host 0.0.0.0 --port 5000 > no_git_oic/api_server.log 2>&1 &
ps -ef | grep api_server
lsof -i :5000


curl -X 'POST' \
  'http://127.0.0.1:5000/get-docs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "CaTiO3解释?",
    "table_name": "mp_database",
    "complicated_question": false,
    "tags": []
  }'
"""
