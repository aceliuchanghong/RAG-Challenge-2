import gradio as gr
import requests
import os
import fitz
from termcolor import colored
import json
from typing import List, Dict, Any

css = """
    footer {
        visibility: hidden;
    }
"""


def judge_files(file_input_list):
    """ """
    un_readable = []
    readable = []

    if not file_input_list:
        return un_readable, readable

    for file_obj in file_input_list:
        file_path = file_obj.name
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".image",
        ]:
            un_readable.append(file_path)
        elif file_extension in [".docx", ".txt", ".md", ".pptx"]:
            readable.append(file_path)
        elif file_extension == ".pdf":
            try:
                with fitz.open(file_path) as doc:
                    is_text_present = False
                    for page in doc:
                        if page.get_text().strip():
                            is_text_present = True
                            break
                    if is_text_present:
                        readable.append(file_path)
                    else:
                        un_readable.append(file_path)
            except Exception as e:
                print(f"Could not process PDF '{file_path}': {e}")
                un_readable.append(file_path)
        else:
            un_readable.append(file_path)

    # print(colored(f"Unreadable files: {un_readable}", "light_yellow"))
    # print(colored(f"Readable files: {readable}", "light_yellow"))
    return un_readable, readable


def deal_readable_file_list(
    readable_file_list, table_name, chunk_size, chunk_overlap, tags_input
):
    api_base_url = "http://127.0.0.1:5000"
    api_url = f"{api_base_url}/deal-files/"
    headers = {"Content-Type": "application/json"}

    # 用于记录处理结果
    successful_files = []
    failed_files = []

    print(f"\n--- ⚡⚡开始同步处理 {len(readable_file_list)} 个文件⚡⚡ ---")

    for index, file_path in enumerate(readable_file_list):
        print(f"[{index + 1}/{len(readable_file_list)}] 正在处理文件: {file_path}...")

        payload = {
            "input_path": file_path,
            "table_name": table_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "tags": tags_input,
            "ser_tab": False,
        }

        try:
            response = requests.post(
                api_url, headers=headers, data=json.dumps(payload), timeout=1200
            )

            response.raise_for_status()

            result = response.json()
            print(f"  ✅ 成功: {file_path} 处理完成。")
            print(f"     服务器响应: {result}")
            successful_files.append({"file": file_path, "response": result})

        except requests.exceptions.RequestException as e:
            error_message = f"请求 API 时发生网络错误: {e}"
            print(f"  ❌ 失败: {error_message}")
            failed_files.append({"file": file_path, "error": error_message})

        except Exception as e:
            error_detail = "未知错误"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            else:
                error_detail = str(e)

            error_message = f"处理文件时发生错误。状态码: {getattr(e.response, 'status_code', 'N/A')}, 详情: {error_detail}"
            print(f"  ❌ 失败: {error_message}")
            failed_files.append({"file": file_path, "error": error_message})

    print(f"--- ⚡⚡结束同步处理 {len(readable_file_list)} 个文件⚡⚡ ---")

    return {
        "successful_readable_files": successful_files,
        "failed_readable_files": failed_files,
    }


def deal_unreadable_file_list(
    un_readable_file_list: List[str],
    table_name: str,
    chunk_size: int,
    chunk_overlap: int,
    tags_input: List[str],
    ser_tab: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    循环处理无法直接读取的文件列表，通过异步上传接口进行处理。

    Args:
        un_readable_file_list (List[str]): 需要处理的文件路径列表。
        table_name (str): 数据要存入的 LanceDB 表名。
        chunk_size (int): 文本分块的大小。
        chunk_overlap (int): 文本分块的重叠大小。
        tags_input (List[str]): 与文件关联的标签列表。
        ser_tab (bool): 是否序列化表格。

    Returns:
        Dict[str, List[Dict[str, Any]]]: 包含成功和失败文件信息的字典。
    """
    api_base_url = "http://127.0.0.1:5000"
    api_url = f"{api_base_url}/upload-and-process-async/"

    # 用于记录处理结果
    successful_files = []
    failed_files = []

    print(f"\n--- ⚡⚡开始异步处理 {len(un_readable_file_list)} 个文件⚡⚡ ---")

    for index, file_path in enumerate(un_readable_file_list):
        print(
            f"[{index + 1}/{len(un_readable_file_list)}] 正在上传文件: {file_path}..."
        )

        # 准备 multipart/form-data 请求体
        # requests 会自动处理 Content-Type
        data_payload = {
            "table_name": table_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "ser_tab": str(ser_tab),  # FastAPI 会自动将 'True'/'true' 字符串转为布尔值
        }
        if tags_input:
            data_payload["tags"] = tags_input

        try:
            # 以二进制模式打开文件
            with open(file_path, "rb") as f:
                files_payload = {"files": (os.path.basename(file_path), f)}

                # 发送 POST 请求
                response = requests.post(
                    api_url, data=data_payload, files=files_payload, timeout=1200
                )

                # 检查请求是否成功 (状态码 2xx)
                response.raise_for_status()

                result = response.json()
                print(f"  ✅ 成功: {file_path} 上传成功，后台任务已启动。")
                print(f"      服务器响应: {result}")
                successful_files.append({"file": file_path, "response": result})

        except FileNotFoundError:
            error_message = f"文件未找到: {file_path}"
            print(f"  ❌ 失败: {error_message}")
            failed_files.append({"file": file_path, "error": error_message})
        except requests.exceptions.RequestException as e:
            error_message = f"请求 API 时发生网络错误: {e}"
            print(f"  ❌ 失败: {error_message}")
            failed_files.append({"file": file_path, "error": error_message})
        except Exception as e:
            # 更详细的错误处理
            error_detail = "未知错误"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail", e.response.text)
                except json.JSONDecodeError:
                    error_detail = e.response.text
            else:
                error_detail = str(e)

            status_code = getattr(e.response, "status_code", "N/A")
            error_message = (
                f"处理文件时发生错误。状态码: {status_code}, 详情: {error_detail}"
            )
            print(f"  ❌ 失败: {error_message}")
            failed_files.append({"file": file_path, "error": error_message})

    print(f"--- ⚡⚡结束异步处理 {len(un_readable_file_list)} 个文件⚡⚡ ---")

    return {
        "successful_unreadable_files": successful_files,
        "failed_unreadable_files": failed_files,
    }


def deal_upload_files(
    file_input_list, table_name, chunk_size, chunk_overlap, tags_input
):
    """处理上传的文件"""

    un_readable_file_list, readable_file_list = judge_files(file_input_list)

    eng_table_name = table_name_dict[table_name]

    result_dict = {}

    if len(readable_file_list) > 0:
        try:
            readable_file_result_dict = deal_readable_file_list(
                readable_file_list,
                eng_table_name,
                chunk_size,
                chunk_overlap,
                tags_input,
            )
            result_dict.update(readable_file_result_dict)
        except Exception as e:
            print(colored(f"ERR:deal_readable_file_list:{e}", "red"))

    if len(un_readable_file_list) > 0:
        try:
            un_readable_file_result_dict = deal_unreadable_file_list(
                un_readable_file_list,
                eng_table_name,
                chunk_size,
                chunk_overlap,
                tags_input,
            )
            result_dict.update(un_readable_file_result_dict)
        except Exception as e:
            print(colored(f"ERR:deal_unreadable_file_list:{e}", "red"))

    return json.dumps(result_dict, ensure_ascii=False, indent=2)


def get_task_status(task_id: str):
    """
    查询指定任务ID的状态。

    Args:
        task_id (str): 任务的唯一标识符。

    Returns:
        dict: 包含任务状态、结果或错误信息的字典。

    Raises:
        requests.exceptions.RequestException: 网络请求失败时抛出。
        ValueError: 当返回状态码为404或其他错误时抛出。
    """
    base_url = "http://127.0.0.1:5000"
    url = f"{base_url}/status/{task_id}"

    try:
        response = requests.get(url)

        if response.status_code == 404:
            raise ValueError("任务未找到")
        elif not response.ok:
            raise ValueError(f"请求失败: {response.status_code}, {response.text}")

        return json.dumps(response.json(), ensure_ascii=False, indent=2)

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"网络请求出错: {e}")


def create_app():
    with gr.Blocks(theme=gr.themes.Soft(), title="研发知识文件上传", css=css) as demo:
        gr.Markdown("## 扫描版文件上传与异步处理")

        with gr.Row():
            with gr.Column():
                file_input_list = gr.File(
                    label="点击上传文件(pptx,docx 文件建议转为扫描版 pdf 处理)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md", ".pptx", "image"],
                )
                file_input_list.GRADIO_CACHE = file_default_path

                table_name = gr.Dropdown(
                    choices=key_list,
                    value=key_list[3],
                    label="库名",
                    info="数据存储到哪个库",
                )

                with gr.Accordion("参数设置", open=False):
                    chunk_size = gr.Slider(
                        minimum=300,
                        maximum=400,
                        step=10,
                        value=300,
                        label="分块大小",
                        info="不建议修改",
                    )
                    chunk_overlap = gr.Slider(
                        minimum=50,
                        maximum=60,
                        step=1,
                        value=50,
                        label="分块重叠",
                        info="不建议修改",
                    )
                    tags_input = gr.Dropdown(
                        label="标签 (输入后按 Enter 添加)",
                        info="例如: NPD2308",
                        multiselect=True,
                        choices=[],
                        allow_custom_value=True,
                    )
                submit_btn = gr.Button("🚀 开始上传并处理")
                clear_button = gr.ClearButton(value="🧹 恢复默认界面")

            with gr.Column():
                with gr.Accordion("进程查询", open=False):
                    uuid_textbox = gr.Textbox(
                        label="输入进程号-task_id",
                        info="eg:6ef43dd1-e471-447b-9058-5377fa6fcdeb",
                    )
                    query_progress_btn = gr.Button("查询进程")
                    progress = gr.Textbox(label="异步处理过程展示", interactive=False)
                output = gr.Textbox(label="结果", interactive=False)

        submit_btn.click(
            deal_upload_files,
            inputs=[file_input_list, table_name, chunk_size, chunk_overlap, tags_input],
            outputs=[output],
        )
        query_progress_btn.click(
            get_task_status, inputs=[uuid_textbox], outputs=[progress]
        )
        clear_button.add(
            [
                file_input_list,
                table_name,
                chunk_size,
                chunk_overlap,
                tags_input,
                output,
                uuid_textbox,
                progress,
            ]
        )
    return demo


if __name__ == "__main__":
    """
    uv run file_upload_gradio_server.py
    nohup uv run file_upload_gradio_server.py > no_git_oic/file_upload_gradio_server.log 2>&1 &

    ps -ef | grep file_upload_gradio_server
    """

    table_name_dict = {
        "研发知识库": "mlcc_database",
        "MLCC知识库": "mp_database",
        "火炬行政库": "torch_database",
        "默认知识库": "file_chunks",
    }
    key_list = list(table_name_dict.keys())

    file_default_path = os.path.join("no_git_oic/people_upload", "new_files")
    os.makedirs(file_default_path, exist_ok=True)

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=20002,
        share=False,
    )
