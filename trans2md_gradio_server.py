import os
import gradio as gr
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread
from pdf2image import convert_from_path
import tempfile
import base64
from io import BytesIO
import time
import shutil


def process_tags(content: str) -> str:
    """将特殊标签替换为 HTML 实体，防止其被渲染为 HTML。"""
    content = content.replace("<img>", "<img>")
    content = content.replace("</img>", "</img>")
    content = content.replace("<watermark>", "<watermark>")
    content = content.replace("</watermark>", "</watermark>")
    content = content.replace("<page_number>", "<page_number>")
    content = content.replace("</page_number>", "</page_number>")
    content = content.replace("<signature>", "<signature>")
    content = content.replace("</signature>", "</signature>")
    return content


def preview_file(file):
    """生成文件预览（支持 PDF 和图像）。"""
    if file is None:
        return None
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".pdf":
        # 转换 PDF 第一页为图像
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, "document.pdf")
            shutil.copy(file_path, temp_pdf_path)
            images = convert_from_path(
                temp_pdf_path, dpi=150, first_page=1, last_page=1
            )
            if images:
                return images[0].convert("RGB").resize((512, 512))
    elif file_ext in [".png", ".jpg", ".jpeg"]:
        # 直接返回图像
        return Image.open(file_path).convert("RGB").resize((512, 512))
    return None


def encode_image(image: Image) -> str:
    """将图像编码为 base64 字符串。"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def stream_request(
    messages: list[dict],
    model_name: str,
    max_tokens: int = 8000,
    temperature: float = 0.0,
):
    """
    从 OCR 模型流式生成文本，基于包含图像和文本的消息。

    参数:
        messages: 包含角色和内容的字典列表
        model_name: 模型名称（未使用，仅为兼容性保留）
        max_tokens: 生成的最大 token 数量
        temperature: 生成的温度（未使用，模型以确定性方式运行）

    返回:
        str: 生成的文本片段
    """
    # 从消息中提取图像和文本
    for message in messages:
        if message["role"] == "user":
            content = message["content"]
            image_data = None
            text_prompt = ""

            for item in content:
                if item["type"] == "image_url":
                    # 解码 base64 图像
                    image_url = item["image_url"]["url"]
                    if image_url.startswith("data:image/jpeg;base64,"):
                        image_base64 = image_url.split(",")[1]
                        image_bytes = base64.b64decode(image_base64)
                        image_data = Image.open(BytesIO(image_bytes))
                elif item["type"] == "text":
                    text_prompt = item["text"]

            if image_data is not None:
                # 格式化消息以符合模型预期格式
                formatted_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_data},
                            {"type": "text", "text": text_prompt},
                        ],
                    },
                ]

                # 应用聊天模板以正确格式化输入
                text = processor.apply_chat_template(
                    formatted_messages, tokenize=False, add_generation_prompt=True
                )

                # 处理格式化文本和图像
                inputs = processor(
                    text=[text], images=[image_data], padding=True, return_tensors="pt"
                )

                # 将输入移动到与模型相同的设备
                inputs = {
                    k: v.to(model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                # 设置流式处理
                streamer = TextIteratorStreamer(
                    tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
                )

                generation_kwargs = {
                    **inputs,
                    "streamer": streamer,
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                }

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                for new_text in streamer:
                    yield new_text

                thread.join()
                return

    yield ""


def convert_to_markdown_stream(
    images: Image, model_name, max_gen_tokens, with_img_desc: bool = False
):
    """
    生成器函数，逐页处理图像并流式生成 Markdown 转换结果。

    参数:
        images: 图像列表
        model_name: 模型名称
        max_gen_tokens: 每页最大生成 token 数
        with_img_desc: 是否包含图像描述

    返回:
        生成的 Markdown 内容
    """
    # 创建 PDF 转 Markdown 的系统提示
    if with_img_desc:
        user_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    else:
        user_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

    full_markdown_content = ""

    for i, image in enumerate(images):
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
            },
            {"type": "text", "text": user_prompt},
        ]

        messages = [{"role": "user", "content": content}]

        # 流式处理单个页面
        page_content = ""
        try:
            for chunk in stream_request(
                messages=messages,
                model_name=model_name,
                max_tokens=max_gen_tokens,
            ):
                page_content += chunk
                current_total = (
                    full_markdown_content
                    + f"### 第 {i + 1} 页，共 {len(images)} 页\n"
                    + page_content
                )
                yield current_total

            full_markdown_content += (
                f"### 第 {i + 1} 页，共 {len(images)} 页\n" + page_content
            )

        except Exception as e:
            return f"错误：{e}"


def process_document(file, max_tokens, with_img_desc: bool = False):
    """
    处理上传的文档（PDF 或图像）并转换为 Markdown。

    参数:
        file: 上传的文件对象
        max_tokens: 每页最大 token 数
        with_img_desc: 是否包含图像描述

    返回:
        生成的 Markdown 内容
    """
    if file is None:
        return "请先上传文件。"

    try:
        file_path = file.name  # 获取文件路径
        file_ext = os.path.splitext(file_path)[1].lower()  # 获取文件扩展名

        if file_ext == ".pdf":
            # 处理 PDF 文件
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, "document.pdf")
                shutil.copy(file_path, temp_pdf_path)

                # 将 PDF 转为图片
                images = convert_from_path(temp_pdf_path, dpi=150)
                images = [image.convert("RGB").resize((2048, 2048)) for image in images]

                # 处理每页
                for result in convert_to_markdown_stream(
                    images, model_dir, max_tokens, with_img_desc
                ):
                    yield process_tags(result)

        elif file_ext in [".png", ".jpg", ".jpeg"]:
            # 处理图片文件
            image = Image.open(file_path).convert("RGB").resize((2048, 2048))

            # 处理单张图片
            for result in convert_to_markdown_stream(
                [image], model_dir, max_tokens, with_img_desc
            ):  # 传入列表以保持接口一致
                yield process_tags(result)

        else:
            return "不支持的文件格式，仅支持 PDF 或图片（PNG/JPG/JPEG）。"

    except Exception as e:
        yield f"处理文档时出错：{str(e)}"


css = """
    #page_info_html {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;  /* 确保与按钮行高度一致 */
        margin: 0 12px;  /* 左右增加边距以居中 */
    }
    
    #page_info_box {
        padding: 8px 20px;
        font-size: 16px;
        border: 1px solid #bbb;
        border-radius: 8px;
        background-color: #f8f8f8;
        text-align: center;
        min-width: 80px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    #markdown_output {
        min-height: 800px;
        max-height: 1000px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        user-select: text; /* 确保文本可被选择 */
        -webkit-user-select: text; /* 兼容 Safari */
        -moz-user-select: text; /* 兼容 Firefox */
        -ms-user-select: text; /* 兼容 Edge */
    }
    
    footer {
        visibility: hidden;
    }
    
    #preview_output {
        max-height: 500px;
        overflow: auto;
        margin-top: 10px;
    }
"""


def create_app():
    """创建 Gradio 应用程序界面。"""
    with gr.Blocks(title="📄Markdown Converter", theme="ocean", css=css) as demo:
        gr.HTML(
            """
        <div class="title" style="text-align: center">
            <h1>PDF & Image 's Markdown Converter</h1>
            <p style="font-size: 1.1em; color: #6b7280; margin-bottom: 0.6em;">
               将文档转换为结构化 Markdown 格式，具备智能内容识别和语义标记功能。
            </p>
        </div>
        """
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="上传图片或 PDF 文档",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    height=200,
                )
                preview_output = gr.Image(
                    label="文件预览",
                    height=350,
                    visible=True,
                )
                max_tokens_slider = gr.Slider(
                    minimum=1024,
                    maximum=8192,
                    value=4096,
                    step=512,
                    label="每页最大 Token 数",
                    info="为每页生成的最大新 Token 数量。",
                )
                with_img_desc_checkbox = gr.Checkbox(
                    label="包含图片描述",
                    value=False,
                    info="如果启用，模型将在输出中包含图片描述。",
                )
                with gr.Row():
                    extract_btn = gr.Button(
                        "转换为 Markdown", variant="primary", size="lg"
                    )
                    clear_btn = gr.ClearButton(variant="secondary", size="lg")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("格式化输出"):
                        output_text = gr.Markdown(
                            label="格式化模型预测结果",
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False},
                            ],
                            line_breaks=True,
                            show_copy_button=True,
                            height=800,
                            container=True,
                            elem_id="markdown_output",
                        )
                    with gr.TabItem("原始 Markdown"):
                        raw_output_text = gr.Textbox(
                            label="原始 Markdown 内容",
                            lines=35,
                            interactive=True,
                            max_lines=3000,
                            show_copy_button=True,
                        )

        extract_btn.click(
            fn=process_document,
            inputs=[file_input, max_tokens_slider, with_img_desc_checkbox],
            concurrency_limit=4,
            outputs=output_text,
        )
        file_input.change(
            fn=preview_file,
            inputs=file_input,
            outputs=preview_output,
        )
        clear_btn.add([file_input, preview_output, output_text, with_img_desc_checkbox])

        output_text.change(
            fn=lambda x: x,
            inputs=output_text,
            outputs=raw_output_text,
        )
    return demo


if __name__ == "__main__":
    """
    安装依赖：
    uv pip install flash-attn gradio transformers pdf2image accelerate torch torchvision vllm

    运行服务：
    uv run trans2md_gradio_server.py
    nohup uv run trans2md_gradio_server.py > no_git_oic/trans2md_gradio_server.log &

    查看进程：
    ps -ef | grep trans2md_gradio_server
    
    移除注释：
    uv run z_utils/remove_comments.py \
        --input trans2md_gradio_server.py \
        --output 00.py
    """
    model_dir = "/mnt/data/llch/Models/Nanonets-OCR-s"

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("模型加载成功！")

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=5001)
