from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import duckdb
import shutil
import json
import uuid
import tempfile

from core import query_documents, process_files_in_pipeline, get_std_md


root_path = Path.cwd()
DB_FILE = root_path / "tasks.duckdb"


# --- DuckDB 初始化和辅助函数 ---
def init_db():
    """初始化数据库，创建 tasks 表（如果不存在）。"""
    with duckdb.connect(str(DB_FILE)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY,
                task_data JSON
            )
        """
        )
    print(f"DuckDB 数据库已在 {DB_FILE} 初始化。")


def db_execute(query: str, params: tuple = ()):
    """执行数据库写操作。"""
    with duckdb.connect(str(DB_FILE)) as con:
        con.execute(query, params)


def db_fetchone(query: str, params: tuple = ()):
    """执行数据库读操作并返回一条记录。"""
    with duckdb.connect(str(DB_FILE)) as con:
        return con.execute(query, params).fetchone()


def get_task_from_db(task_id: str) -> Optional[Dict[str, Any]]:
    """从数据库中获取任务。"""
    result = db_fetchone("SELECT task_data FROM tasks WHERE task_id = ?", (task_id,))
    return json.loads(result[0]) if result else None


def update_task_in_db(task_id: str, updates: Dict[str, Any]):
    """更新数据库中的任务数据。"""
    current_task = get_task_from_db(task_id)
    if current_task:
        current_task.update(updates)
        db_execute(
            "UPDATE tasks SET task_data = ? WHERE task_id = ?",
            (json.dumps(current_task), task_id),
        )


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    init_db()
    yield
    # Code to run on shutdown (e.g., close database connections)
    print("Application shutting down.")


# -----------数据库结束---------------

app = FastAPI(
    title="RAG API",
    description="一个用于文档 rag 处理的 API。",
    version="1.0.0",
    lifespan=lifespan,
)


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


async def run_processing_pipeline(
    task_id: str,
    input_files_paths: List[str],
    output_dir: str,
    table_name: str,
    chunk_size: int,
    chunk_overlap: int,
    tags: List[str],
    ser_tab: bool,
):
    """这个函数将在后台运行，执行完整的处理流程，并更新DuckDB中的状态。"""
    try:
        update_task_in_db(task_id, {"status": "converting_to_markdown"})
        print(f"任务 {task_id}: 开始Markdown转换...")

        input_data_for_md = (
            input_files_paths[0] if len(input_files_paths) == 1 else input_files_paths
        )
        markdown_file_path = await get_std_md(
            input_data=input_data_for_md, output_dir=output_dir
        )

        update_task_in_db(
            task_id,
            {"markdown_path": markdown_file_path, "status": "embedding_and_storing"},
        )
        print(
            f"任务 {task_id}: Markdown转换完成，位于 {markdown_file_path}。开始向量化..."
        )

        result = await process_files_in_pipeline(
            input_path=markdown_file_path,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tags=tags,
            ser_tab=ser_tab,
            root_path=root_path,
        )

        update_task_in_db(task_id, {"status": "completed", "result": result})
        print(f"任务 {task_id}: 处理成功完成。")

    except Exception as e:
        print(f"任务 {task_id}: 处理失败。错误: {e}")
        update_task_in_db(task_id, {"status": "failed", "error": str(e)})
    finally:
        temp_dir = Path(input_files_paths[0]).parent
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"任务 {task_id}: 已清理临时目录 {temp_dir}")


@app.post("/upload-and-process-async/", summary="上传文件并启动后台处理任务")
async def upload_and_process_api(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="一个或多个要处理的文件。"),
    table_name: str = Form("file_chunks", description="数据要存入的 LanceDB 表名。"),
    chunk_size: int = Form(300, description="文本分块的大小。"),
    chunk_overlap: int = Form(50, description="文本分块的重叠大小。"),
    tags: Optional[List[str]] = Form(
        None, description="与文件关联的标签列表。例如：--tags test1 --tags test2"
    ),
    ser_tab: bool = Form(True, description="是否序列化表格"),
):
    """接收上传文件，立即返回一个任务ID，并在后台异步执行完整的处理流水线。"""
    task_id = str(uuid.uuid4())

    save_dir = root_path / "no_git_oic/save_files_async/"
    save_dir.mkdir(parents=True, exist_ok=True)

    input_files_paths = []
    original_filenames = []
    for file in files:
        file_path = save_dir / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_files_paths.append(str(file_path))
            original_filenames.append(file.filename)
        finally:
            file.file.close()

    # 在数据库中创建初始任务记录
    initial_task_data = {
        "task_id": task_id,
        "status": "processing_started",
        "original_filenames": original_filenames,
    }
    db_execute(
        "INSERT INTO tasks (task_id, task_data) VALUES (?, ?)",
        (task_id, json.dumps(initial_task_data)),
    )

    # 将核心处理函数作为后台任务添加
    background_tasks.add_task(
        run_processing_pipeline,
        task_id=task_id,
        input_files_paths=input_files_paths,
        output_dir=str(root_path / "no_git_oic/save_md/"),
        table_name=table_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tags=tags if tags is not None else [],
        ser_tab=ser_tab,
    )

    return {"message": "文件上传成功，处理任务已在后台启动。", "task_id": task_id}


@app.get("/status/{task_id}", summary="查询后台任务的状态")
async def get_task_status_api(task_id: str):
    """根据任务ID从DuckDB查询任务的当前状态、结果或错误信息。"""
    task = get_task_from_db(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务未找到")
    return task


@app.post("/upload-and-process/", summary="上传文件并转换为Markdown-同步")
async def upload_to_markdown_api(
    files: List[UploadFile] = File(
        ..., description="要上传并转换为Markdown的一个或多个文件。"
    ),
    output_dir: Optional[str] = Form(
        "no_git_oic/save_md/", description="Markdown文件的输出目录。"
    ),
):
    """
    **（同步阻塞接口）**
    接收一个或多个上传的文件，处理并保存为 Markdown 文件，然后返回路径。
    """
    try:

        input_for_get_std_md = files[0] if len(files) == 1 else files
        # 调用核心处理函数
        markdown_file_path = await get_std_md(
            input_data=input_for_get_std_md, output_dir=output_dir
        )

        # 提取所有上传文件的文件名
        original_filenames = [file.filename for file in files]

        return {
            "message": "文件已成功转换为Markdown。",
            "markdown_file_path": markdown_file_path,
            "processed_filenames": original_filenames,
        }

    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获处理过程中可能发生的任何错误
        raise HTTPException(status_code=500, detail=f"文件转换过程中发生错误: {e}")


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

# 同步处理
curl -X 'POST' \
  'http://127.0.0.1:5000/upload-and-process/' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/pdf_images_NPD2308工艺文件/page_16.png'

# 异步处理pdf
curl -X POST "http://127.0.0.1:5000/upload-and-process-async/" \
     -F "files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/22.pdf" \
     -F "table_name=my_test_table"

curl http://127.0.0.1:5000/status/0507a154-3f2c-4420-95a0-41dcc220397b
"""
