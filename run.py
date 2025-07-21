import click
import os
from termcolor import colored
from pathlib import Path
import json
import shutil
from openai import OpenAI
import time
from typing import Optional
import asyncio

from z_utils.hash_x import compute_mdhash_id
from code.pipeline import Pipeline
from code.common import extract_hash, DocumentType
from code.prompt import PROMPT_WITH_EVIDENCE, PROMPT_GENERAL


root_path = Path.cwd()


@click.group()
def cli():
    pass


def _process_file(file_path: Path, output_dir: str):
    """
    处理单个文件的核心逻辑 读取、计算哈希、保存md文件和原始文件。

    Args:
        file_path (Path): 要处理的文件的路径.
        output_dir (str): md文件的输出目录.
    """
    try:
        pipeline = Pipeline(root_path)
        click.echo(colored(f"Reading file: {file_path}", "yellow"))
        content = pipeline.read_file(file_path)
        mdhash_id = compute_mdhash_id(content)
        output_path = Path(output_dir) / ("md_" + mdhash_id + ".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(
            colored(
                f"Successfully processed and saved content to {output_path}",
                "green",
            )
        )
        original_path = Path(output_dir).parent / "original"
        original_path.mkdir(exist_ok=True)
        dest_file = original_path / f"{mdhash_id}{file_path.suffix}"
        shutil.copy(file_path, dest_file)
    except ValueError as e:
        click.echo(colored(f"Error processing {file_path.name}: {e}", "red"))
    except Exception as e:
        click.echo(
            colored(
                f"[_process_file] An unexpected error occurred with {file_path.name}: {e}",
                "red",
            )
        )


async def _process_and_chunk_file(
    md_file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    output_dir: str,
    tags: tuple[str],
    ser_tab: bool,
):
    """
    处理单个md文件的切片核心逻辑。

    Args:
        md_file_path (Path): 要处理的md文件的路径.
        chunk_size (int): 切片大小.
        chunk_overlap (int): 切片重叠大小.
        output_dir (str): 切片结果(.jsonl)的输出目录.
    """
    try:
        pipeline = Pipeline(root_path)
        click.echo(colored(f"Chunking markdown file: {md_file_path.name}", "yellow"))
        chunks = await pipeline.chunk_md_file(
            str(md_file_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tags=tags,
            ser_tab=ser_tab,
        )
        file_hash = extract_hash(md_file_path.name)
        output_path = Path(output_dir) / ("chunked_" + file_hash + ".jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json_line = json.dumps(chunks, ensure_ascii=False)
            f.write(json_line + "\n")
        click.echo(
            colored(
                f"Successfully chunked file and saved to {output_path}",
                "green",
            )
        )
    except Exception as e:
        click.echo(
            colored(
                f"[chunk_markdown] An unexpected error occurred with {md_file_path.name}: {e}",
                "red",
            )
        )


@cli.command()
@click.option("--models", default="nanonets/Nanonets-OCR-s", help="模型")
@click.option("--source", default="ms", type=click.Choice(["ms", "hf"]), help="下载源")
def download_models(models, source):
    """
    Download required models from specified source.

    #### 主要测试了:
    - nanonets/Nanonets-OCR-s
    - monkeyOCR
    - MinerU
    - ChatDOC/OCRFlux-3B
    - rapidai(ocr&table)
    """
    click.echo(f"Downloading {models} models from {source}...")

    # 设置输出路径
    output_path = "/mnt/data/llch/Models/" + models.split("/")[1]
    os.makedirs(output_path, exist_ok=True)

    if source == "hf":
        # ChatDOC/OCRFlux-3B
        from huggingface_hub import snapshot_download

        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    else:
        from modelscope import snapshot_download

    # 执行下载
    model_dir = snapshot_download(
        repo_id=models,
        local_dir=output_path,
        # resume_download=True,
    )

    click.echo(f"Downloaded {models} models successfully to {model_dir}")


@cli.command()
@click.option("--file-path", help="待读取的文件或文件夹")
@click.option("--output", default="output/md", help="输出目录")
def read_files(file_path, output):
    """读取文件或文件夹下面所有特定后缀文件转为md"""
    path = Path(file_path)
    allowed_suffixes = {
        f".{ext}" for doc_type in DocumentType for ext in doc_type.value
    }
    if not path.exists():
        click.echo(colored(f"Error: Path not found at '{file_path}'", "red"))
        return
    if path.is_file():
        if path.suffix in allowed_suffixes:
            _process_file(path, output)
        else:
            click.echo(
                colored(
                    f"Skipping file: Unsupported file type '{path.suffix}' for {path.name}",
                    "magenta",
                )
            )
    elif path.is_dir():
        click.echo(colored(f"Reading allowed files from directory: {path}", "cyan"))
        processed_count = 0
        for item in path.iterdir():  # 不进入子目录
            if item.is_file():
                if item.suffix in allowed_suffixes:
                    _process_file(item, output)
                    processed_count += 1
                else:
                    click.echo(
                        colored(
                            f"Skipping file: Unsupported file type '{item.suffix}' for {item.name}",
                            "magenta",
                        )
                    )
        if processed_count == 0:
            click.echo(colored(f"No processable files found in {path}", "yellow"))
        else:
            click.echo(
                colored(
                    f"Finished processing directory. Total files processed: {processed_count}",
                    "green",
                )
            )
    else:
        click.echo(
            colored(
                f"Error: Path '{file_path}' is not a valid file or directory.", "red"
            )
        )


@cli.command()
@click.option("--md-file-path", help="MD文件或者文件夹")
@click.option("--chunk-size", default=300, type=int, help="切块大小")
@click.option("--chunk-overlap", default=50, type=int, help="切块重合大小")
@click.option("--output", default="output/chunked_md", help="输出目录")
@click.option("--tags", multiple=True, help="添加 tags")
@click.option("--ser-tab", is_flag=True, default=False, help="开启表格处理")
def chunk_markdown(md_file_path, chunk_size, chunk_overlap, output, tags, ser_tab):
    """
    将一个 'md_{hash}.md' 文件或一个目录中所有符合该格式的文件进行切片。
    """
    path = Path(md_file_path)
    if not path.exists():
        click.echo(colored(f"Error: Path not found at '{md_file_path}'", "red"))
        return
    if path.is_file():
        try:
            # 验证文件名格式是否正确
            extract_hash(path.name)
            asyncio.run(
                _process_and_chunk_file(
                    path, chunk_size, chunk_overlap, output, tags, ser_tab
                )
            )
        except ValueError as e:
            # 文件名格式不正确，进行提示并跳过
            click.echo(colored(f"Skipping file: {path.name}. Reason: {e}", "magenta"))
    elif path.is_dir():
        click.echo(colored(f"Chunking valid files from directory: {path}", "cyan"))
        processed_count = 0
        for item in path.iterdir():
            if item.is_file():
                try:
                    # 验证文件名格式是否正确，不正确则跳过
                    extract_hash(item.name)
                    asyncio.run(
                        _process_and_chunk_file(
                            item, chunk_size, chunk_overlap, output, tags, ser_tab
                        )
                    )
                    processed_count += 1
                except ValueError:
                    pass  # 静默跳过
        if processed_count == 0:
            click.echo(
                colored(
                    f"No files with format 'md_{{hash}}.md' found in {path}", "yellow"
                )
            )
        else:
            click.echo(
                colored(
                    f"Finished processing directory. Total files chunked: {processed_count}",
                    "green",
                )
            )
    else:
        click.echo(
            colored(
                f"Error: Path '{md_file_path}' is not a valid file or directory.", "red"
            )
        )


@cli.command()
@click.option("--jsonl-path-or-dir", default="output/chunked_md", help="文件或文件夹")
@click.option("--table-name", default="保存表名")
def save_jsonl(jsonl_path_or_dir, table_name):
    """数据库存储"""
    try:
        pipeline = Pipeline(root_path)
        pipeline.save2lacncedb(
            report_or_reports_dir=jsonl_path_or_dir, table_name=table_name
        )
        click.echo(
            colored(
                f"Successfully saved JSONL files to LanceDB table '{table_name}'",
                "green",
            )
        )
        return table_name
    except Exception as e:
        click.echo(colored(f"[save_jsonl] An unexpected error occurred: {e}", "red"))
        return None


@cli.command()
@click.option("--question", help="输入问题")
@click.option("--table-name", default="file_chunks", help="表名")
@click.option("--complicated-question", is_flag=True, default=False, help="复杂问题")
@click.option("--tags", multiple=True, help="寻找 tags")
def get_docs(
    question: str, table_name: str, complicated_question: bool, tags: tuple[str]
):
    """
    快速检索问题
    """
    try:
        start_time = time.time()
        pipeline = Pipeline(root_path)
        question_related_docs = pipeline.find_question_related_docs(
            question, table_name, complicated_question, tags
        )
        for item in question_related_docs["results"]:
            if "relevance_score" in item:
                del item["relevance_score"]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(colored(f"检索文档耗时: {elapsed_time:.2f}秒", "magenta"))
        if question_related_docs:
            return question_related_docs["results"]
        else:
            click.echo(colored("未找到相关文档", "red"))
            return None
    except Exception as e:
        click.echo(colored(f"\n[get_docs] 发生意外错误: {e}", "red"))
        return None


@cli.command()
@click.option("--question", help="输入问题")
@click.option("--table-name", default="file_chunks", help="表名")
@click.option("--stream", is_flag=True, default=False, help="开启流式输出")
@click.option("--complicated-question", is_flag=True, default=False, help="复杂问题")
@click.option("--tags", multiple=True, help="寻找 tags")
def answer_question(
    question: str,
    table_name: str,
    stream: bool,
    complicated_question: bool,
    tags: tuple[str],
):
    """
    快速回答问题
    """
    llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    try:
        start_time = time.time()
        pipeline = Pipeline(root_path)
        question_related_docs = pipeline.find_question_related_docs(
            question, table_name, complicated_question, tags
        )
        # print(f"bb{question_related_docs}")
        for item in question_related_docs["results"]:
            if "relevance_score" in item:
                del item["relevance_score"]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(colored(f"检索文档耗时: {elapsed_time:.2f}秒", "magenta"))
        messages = []
        question_related_docs = question_related_docs["results"]
        # print(f"{question_related_docs}")
        if question_related_docs:
            final_prompt = PROMPT_WITH_EVIDENCE.format(
                question=question, context=question_related_docs
            )

            messages = [{"role": "user", "content": final_prompt}]
        else:
            click.echo(colored("未找到相关文档，将进行普通回答...", "red"))
            messages = [
                {"role": "user", "content": PROMPT_GENERAL.format(question=question)},
            ]

        # print(f"{messages}")
        start_time = time.time()
        click.echo("=" * 50)
        click.echo(colored(f"Q: {question}", "blue"))
        response = llm.chat.completions.create(
            model="Qwen3",
            messages=messages,
            stream=stream,
            temperature=0.7,
        )

        full_response = ""
        click.echo(colored("A:", "green"), nl=False)
        if stream:
            chunk_buffer = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    chunk_buffer += content
                    full_response += content
                    print(colored(content, "green"), end="", flush=True)
            print()
        else:
            full_response = response.choices[0].message.content
            click.echo(colored(full_response, "green"))
        click.echo("=" * 50)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(colored(f"生成结果耗时: {elapsed_time:.2f}秒", "magenta"))
        return full_response
    except Exception as e:
        click.echo(colored(f"\n发生意外错误: {e}", "red"))
        return None


if __name__ == "__main__":
    """
    uv run run.py download-models

    uv run run.py read-files --file-path no_git_oic/test_files/流式细胞制备方案.pdf
    uv run run.py read-files --file-path no_git_oic/test_files/

    uv run run.py chunk-markdown --md-file-path output/md/md_2a756c2048842968844b3d504cfd33b0.md --tags test1 --tags test2
    uv run run.py chunk-markdown --md-file-path output/md/

    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_2a756c2048842968844b3d504cfd33b0.jsonl
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md --table-name new_test

    uv run run.py get-docs --question "流式细胞制备如何操作?" --complicated-question --tags test1

    uv run run.py answer-question --question "流式细胞制备如何操作?" --stream
    uv run run.py answer-question --question "就三国演义小说介绍一下庞统的生平" --stream --table-name sanguo --complicated-question

    uv run run.py read-files --file-path no_git_oic/test_files/三国演义.docx
    uv run run.py chunk-markdown --md-file-path output/md/md_c6f5b8c6fc281b49f3b50cc778c5cecc.md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_c6f5b8c6fc281b49f3b50cc778c5cecc.jsonl --table-name sanguo
    """
    cli()
