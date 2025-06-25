import click
import os
from termcolor import colored
from pathlib import Path
import json
from code.pipeline import Pipeline
import shutil
from openai import OpenAI
import time

from code.common import extract_hash
from z_utils.hash_x import compute_mdhash_id
from code.prompt import PROMPT_WITH_EVIDENCE, PROMPT_GENERAL

root_path = Path.cwd()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--file-path", help="Path to the file to be read.")
@click.option("--output", default="output/md")
def read_files(file_path, output):
    """读取文件转为md"""
    if not Path(file_path).is_file():
        click.echo(colored(f"Error: File not found at {file_path}", "red"))
        return
    try:
        pipeline = Pipeline(root_path)
        click.echo(colored(f"Reading file: {file_path}", "yellow"))
        content = pipeline.read_file(file_path)
        mdhash_id = compute_mdhash_id(content)
        output_path = Path(output) / ("md_" + mdhash_id + ".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(
            colored(
                f"Successfully processed file and saved content to {output_path}",
                "green",
            )
        )
        original_path = Path(output).parent / "original"
        original_path.mkdir(exist_ok=True)
        src_file = Path(file_path)
        dest_file = original_path / f"{mdhash_id}{src_file.suffix}"
        shutil.copy(src_file, dest_file)
        return output_path
    except ValueError as e:
        click.echo(colored(str(e), "red"))
    except Exception as e:
        click.echo(colored(f"[read_files] An unexpected error occurred: {e}", "red"))


@cli.command()
@click.option("--md-file-path", help="MD file path")
@click.option("--chunk-size", default=300, type=int)
@click.option("--chunk-overlap", default=50, type=int)
@click.option("--output", default="output/chunked_md")
def chunk_markdown(md_file_path, chunk_size, chunk_overlap, output):
    """Chunk md file"""
    if not Path(md_file_path).is_file():
        click.echo(colored(f"Error: File not found at {md_file_path}", "red"))
        return
    if not md_file_path.endswith(".md"):
        click.echo(colored("Error: The file must be a markdown (.md) file.", "red"))
        return
    try:
        pipeline = Pipeline(root_path)
        click.echo(colored(f"Chunking markdown file: {md_file_path}", "yellow"))
        chunks = pipeline.chunk_md_file(
            md_file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        file_name = os.path.basename(md_file_path)
        try:
            file_hash = extract_hash(file_name)
        except ValueError as e:
            click.echo(colored(str(e), "red"))
        output_path = Path(output) / ("chunked_" + file_hash + ".jsonl")
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
        return output_path
    except ValueError as e:
        click.echo(colored(str(e), "red"))
    except Exception as e:
        click.echo(
            colored(f"[chunk_markdown] An unexpected error occurred: {e}", "red")
        )


@cli.command()
@click.option("--jsonl-path-or-dir", default="output/chunked_md", help="path or dir")
@click.option("--table-name", default="file_chunks")
def save_jsonl(jsonl_path_or_dir, table_name):
    """save jsonl files to LanceDB"""
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
@click.option("--question")
@click.option("--table-name", default="file_chunks")
@click.option("--stream", is_flag=True, default=False, help="Enable streaming response")
@click.option("--complicated-question", is_flag=True, default=False)
def answer_question(
    question: str, table_name: str, stream: bool, complicated_question: bool
):
    """
    Process question and return answer using the pipeline.
    """
    llm = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

    try:
        start_time = time.time()
        pipeline = Pipeline(root_path)
        question_related_docs = pipeline.find_question_related_docs(
            question, table_name, complicated_question
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(colored(f"检索文档耗时: {elapsed_time:.2f}秒", "magenta"))
        messages = []
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
    uv run run.py read-files --file-path no_git_oic/test_files/流式细胞制备方案.pdf
    uv run run.py chunk-markdown --md-file-path output/md/md_2a756c2048842968844b3d504cfd33b0.md

    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_2a756c2048842968844b3d504cfd33b0.jsonl
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md --table-name new_test

    uv run run.py answer-question --question "流式细胞制备如何操作?" --stream
    uv run run.py answer-question --question "就三国演义小说介绍一下庞统的生平" --stream --table-name sanguo --complicated-question

    uv run run.py read-files --file-path no_git_oic/test_files/三国演义.docx
    uv run run.py chunk-markdown --md-file-path output/md/md_c6f5b8c6fc281b49f3b50cc778c5cecc.md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_c6f5b8c6fc281b49f3b50cc778c5cecc.jsonl --table-name sanguo
    """
    cli()
