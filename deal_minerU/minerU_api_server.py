from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi import Form
from typing import List, Optional
from pathlib import Path
import shutil
import json
import os
from pathlib import Path
from loguru import logger

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        pass
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        # from mineru.backend.vlm.vlm_analyze import ModelSingleton
        # predictor = ModelSingleton().get_model(backend, model_path, server_url)
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                # predictor=predictor,
                image_writer=image_writer,
                backend=backend,
                server_url=server_url,
            )

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                )

            if f_draw_span_bbox:
                draw_span_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                )

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
    path_list: list[Path],
    output_dir,
    lang="ch",
    backend="pipeline",
    method="auto",
    server_url=None,
    start_page_id=0,
    end_page_id=None,
):
    """
    Parameter description:
    path_list: List of document paths to be parsed, can be PDF or image files.
    output_dir: Output directory for storing parsing results.
    lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
        Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
        Adapted only for the case where the backend is set to "pipeline"
    backend: the backend for parsing pdf:
        pipeline: More general.
        vlm-transformers: More general.
        vlm-sglang-engine: Faster(engine).
        vlm-sglang-client: Faster(client).
        without method specified, pipeline will be used by default.
    method: the method for parsing pdf:
        auto: Automatically determine the method based on the file type.
        txt: Use text extraction method.
        ocr: Use OCR method for image-based PDFs.
        Without method specified, 'auto' will be used by default.
        Adapted only for the case where the backend is set to "pipeline".
    server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    start_page_id: Start page ID for parsing, default is 0
    end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except Exception as e:
        logger.exception(e)


def deal_md_pics(md_content, file_stem, prompt):
    from functools import partial
    from tools import replacer
    import re

    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    replacer_with_prompt = partial(replacer, file_stem=file_stem, prompt=prompt)
    new_md_content = image_pattern.sub(replacer_with_prompt, md_content)

    return new_md_content


app = FastAPI(
    title="LCH Document Processing API",
    description="基于 minerU 的文档解析 API",
    version="1.1.0",
)

root_path = Path.cwd()
TEMP_UPLOAD_DIR = root_path / "no_git_oic/upload_files"
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/trans2md/", summary="将图片或 PDF 转换为 Markdown")
async def trans2md_api(
    files: List[UploadFile] = File(..., description="要处理的图片或 PDF 文件列表。"),
    output_dir: str = Form(
        default="no_git_oic/upload_files_md_output",
        description="保存生成的 Markdown 文件的目录。",
    ),
    backend: str = Form(
        default="vlm-sglang-client", description="用于处理的后端模型。"
    ),
    server_url: Optional[str] = Form(
        default="http://127.0.0.1:30000", description="后端推理服务器的 URL"
    ),
    use_visual_model: bool = Form(
        default=False, description="是否启用视觉模型进行解析。"
    ),
    visual_model_prompt: str = Form(
        default="1. 首先完整描述图片的内容\n2. 然后将图片里面文本以 markdown 格式输出\n3. json输出,不要多余的文字\neg:{'描述':'xx','markdown':'xx'}",
        description="提供给视觉模型的提示词",
    ),
):
    """
    上传一个或多个图片/PDF文件 将其转换为 Markdown 并返回输出目录的路径。

    每个文件都会被单独处理。
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件。")

    saved_file_paths = []
    for one_file in files:
        if not (
            one_file.content_type in ["application/pdf", "image/png", "image/jpeg"]
        ):
            raise HTTPException(
                status_code=400,
                detail=f"无效的文件类型: {one_file.content_type}。仅支持 PDF, PNG, JPG, JPEG 格式。",
            )

        try:
            temp_path = TEMP_UPLOAD_DIR / one_file.filename
            with temp_path.open("wb") as buffer:
                shutil.copyfileobj(one_file.file, buffer)
            saved_file_paths.append(temp_path)
        finally:
            one_file.file.close()

    # 创建输出目录
    output_dir_path = root_path / output_dir
    output_dir_path.mkdir(exist_ok=True)

    try:
        # 将耗时的同步代码（也就是 parse_doc 函数）放到一个独立的线程池中执行，以避免阻塞主事件循环
        await run_in_threadpool(
            parse_doc,
            path_list=saved_file_paths,
            output_dir=str(output_dir_path),
            backend=backend,
            server_url=server_url,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理过程中发生错误：{e}")
    finally:
        # 不管是否成功处理，最后都会删除上传时保存在 TEMP_UPLOAD_DIR 中的原始文件。
        for path in saved_file_paths:
            path.unlink()

    # 构建输出结果，包含路径和对应的 .md 文件内容
    results = []
    for p in saved_file_paths:
        file_stem = p.stem
        md_dir = output_dir_path / file_stem / "vlm"
        md_file_path = md_dir / f"{file_stem}.md"

        # 确保文件存在再读取
        if not md_file_path.exists():
            raise HTTPException(
                status_code=500, detail=f"Markdown 文件未生成: {md_file_path}"
            )

        try:
            with open(md_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if use_visual_model:
                    content = deal_md_pics(content, file_stem, visual_model_prompt)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"无法读取 Markdown 文件内容: {e}"
            )

        results.append({"path": str(md_dir), "content": content})

    """
    {
        "output_files": [
            {
                "path": "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/upload_files_md_output/test_J/vlm",
                "content": "# Markdown 内容...",
            },
            {
                "path": "/mnt/data/llch/RAG-Challenge-2/deal_minerU/no_git_oic/upload_files_md_output/00/vlm",
                "content": "# 另一个 Markdown 内容...",
            },
        ]
    }
    """
    return {"output_files": results}


# 根路径 欢迎接口
@app.get("/")
def read_root():
    return {"message": "欢迎使用文档处理 API 访问 /docs 查看文档。"}


"""
cd deal_minerU
export no_proxy="localhost,127.0.0.1"

uvicorn minerU_api_server:app --host 0.0.0.0 --port 5005
nohup uvicorn minerU_api_server:app --host 0.0.0.0 --port 5005 > no_git_oic/minerU_api_server.log 2>&1 &
ps -ef | grep minerU_api_server
lsof -i :5005

# 单个文件
curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J.png'

curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J.png' \
  -F 'use_visual_model=true'

curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/00.pdf' \
  -F 'use_visual_model=true'

curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/image.png' \
  -F 'use_visual_model=true'
  
# 多个文件
curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J.png' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/00.pdf'

curl -X 'POST' \
  'http://127.0.0.1:5005/trans2md/' \
  -H 'accept: application/json' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J.png' \
  -F 'files=@/mnt/data/llch/RAG-Challenge-2/no_git_oic/00.pdf' \
  -F 'output_dir=no_git_oic/upload_files_md_output' \
  -F 'backend=vlm-sglang-client' \
  -F 'server_url=http://127.0.0.1:30000' \
  -F 'use_visual_model=true' \
  -F 'visual_model_prompt=对图片进行OCR,输出markdown格式'
"""
