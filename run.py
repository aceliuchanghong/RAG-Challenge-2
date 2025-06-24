import click
import os
from termcolor import colored
from pathlib import Path
import json
from code.pipeline import Pipeline
from z_utils.hash_x import compute_mdhash_id

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
        output_name = compute_mdhash_id(content, prefix="md_")
        output_path = Path(output) / (output_name + ".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(
            colored(
                f"Successfully processed file and saved content to {output_path}",
                "green",
            )
        )
        return output_path
    except ValueError as e:
        click.echo(colored(str(e), "red"))
    except Exception as e:
        click.echo(colored(f"An unexpected error occurred: {e}", "red"))


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
        file_hash = file_name[3:-3]
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
        click.echo(colored(f"An unexpected error occurred: {e}", "red"))


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
        click.echo(colored(f"An unexpected error occurred: {e}", "red"))


@cli.command()
@click.option("--question")
def answer_question(question: str) -> list[str]:
    """
    Process question and return answer using the pipeline.
    """
    try:
        pipeline = Pipeline(root_path)
        answer = pipeline.answer_questions(question)
        if answer is None:
            click.echo(colored("Pipeline 没有返回结果。", "red"))
        else:
            print("=" * 50)
            click.echo(colored(f"Q: {question}", "blue"))
            click.echo(colored(f"A: {answer}", "green"))
            print("=" * 50)
        return answer
    except Exception as e:
        click.echo(colored(f"发生意外错误: {e}", "red"))
        return None


if __name__ == "__main__":
    """
    uv run run.py read-files --file-path no_git_oic/test_files/流式细胞制备方案.pdf
    uv run run.py chunk-markdown --md-file-path output/md/md_2a756c2048842968844b3d504cfd33b0.md

    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_2a756c2048842968844b3d504cfd33b0.jsonl
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md --table-name new_test

    uv run run.py answer-question --question "流式细胞制备如何操作?"

    uv run run.py read-files --file-path no_git_oic/test_files/三国演义.docx
    uv run run.py chunk-markdown --md-file-path output/md/md_c6f5b8c6fc281b49f3b50cc778c5cecc.md
    uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_c6f5b8c6fc281b49f3b50cc778c5cecc.jsonl
    """
    cli()
