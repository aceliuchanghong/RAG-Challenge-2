import base64
import os
from typing import Optional
from openai import OpenAI
import re
from typing import Match
from functools import partial


def get_image_description(
    image_path: str,
    prompt: str = "1. 首先完整描述图片的内容\n2. 然后将图片里面文本以 markdown 格式输出\n3. json输出,不要多余的文字\neg:{'描述':'xx','markdown':'xx'}",
    model_name: str = "qwen2.5vl",
    host: str = "http://localhost:11434",
    api_key: str = "no-key-required",
) -> Optional[str]:
    """

    Args:
        image_path (str): 图片的本地文件路径。
        prompt (str): 对模型下达的指令。
        model_name (str): 要使用的 VLM 模型名称，默认为 qwen2.5vl。
        host (str): VLM 服务地址，默认为本地 http://localhost:11434。

    Returns:
        Optional[str]: 模型生成的描述文本，如果失败则返回 None。
    """

    if not os.path.exists(image_path):
        print(f"警告: 图片文件未找到，跳过: {image_path}")
        return None

    # 将图片编码为 base64
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    # 初始化 OpenAI 客户端，指向本地 VLM
    client = OpenAI(base_url=f"{host}/v1", api_key=api_key)

    print(f"正在向本地 VLM 的模型 '{model_name}' 发送请求...")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )

        description = response.choices[0].message.content
        return description.strip()

    except Exception as e:
        print(f"请求 VLM 出错：{e}")
        return None


def replacer(match: Match, file_stem: str, prompt: str) -> str:
    """
    这是一个替换函数 re.sub 会为每个匹配项调用它。
    """
    alt_text = match.group(1)
    image_path = match.group(2)
    original_tag = match.group(0)

    print(f"--- 开始处理匹配项 ---", flush=True)
    print(f"  Alt Text: '{alt_text}'", flush=True)
    print(f"  Image Path: '{image_path}'", flush=True)
    print(f"  Original Tag: {original_tag}", flush=True)

    print(f"发现图片标签，准备为路径 {image_path} 获取描述...", flush=True)

    try:
        full_image_path = os.path.join(
            "no_git_oic", "upload_files_md_output", file_stem, "vlm", image_path
        )
        # model_name = "qwen2.5vl"
        # host = "http://localhost:11434"
        # api_key = "no-key-required"
        model_name = "Qwen2.5-VL-32B-Instruct"
        host = "http://192.168.180.39:6006"
        api_key = "torch-elskenrgvoiserngviopsejrmoief"
        description = get_image_description(
            full_image_path, prompt, model_name, host, api_key
        )
        print(f"成功获取描述: '{description}'", flush=True)

        return f"<image_description>{description}</image_description>"

    except Exception as e:
        print(f"❌ 错误: 处理图片 {image_path} 失败: {e}", flush=True)
        return original_tag
    finally:
        print(f"--- 完成处理匹配项 ---", flush=True)


if __name__ == "__main__":
    """
    export no_proxy="localhost,127.0.0.1,192.168.180.39"
    uv run tools.py
    """
    image_path = "no_git_oic/upload_files_md_output/test_J2/vlm/images/40cd422f3b1b5339c0a7ca0f6b06fe3b127d9bdc3819714b73d6dabea136418c.jpg"
    prompt = "1. 首先完整描述图片的内容\n2. 然后将图片里面文本以 markdown 格式输出\n3. json输出,不要多余的文字\neg:{'描述':'xx','markdown':'xx'}"
    model_name = "Qwen2.5-VL-32B-Instruct"
    host = "http://192.168.180.39:6006"
    host = "http://127.0.0.1:6006"
    api_key = "torch-elskenrgvoiserngviopsejrmoief"
    print(f"{get_image_description(image_path, prompt, model_name, host, api_key)}")

    # aa = "no_git_oic/upload_files_md_output/00/vlm/00.md"
    # with open(aa, "r", encoding="utf-8") as f:
    #     md_content = f.read()

    # prompt = "1. 首先完整描述图片的内容\n2. 然后将图片里面文本以 markdown 格式输出\n3. json输出,不要多余的文字\neg:{'描述':'xx','markdown':'xx'}"
    # print(">>> 开始替换图片标签...", flush=True)
    # image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    # replacer_with_prompt = partial(replacer, file_stem="00", prompt=prompt)
    # new_md_content = image_pattern.sub(replacer_with_prompt, md_content)
    # print("\n>>> 替换完成！", flush=True)

    # print("\n---最终输出内容---")
    # print(f"{new_md_content}")
