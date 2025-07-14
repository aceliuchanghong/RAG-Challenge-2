from openai import OpenAI
import os
import base64


def image_to_base64_data_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded_image}"


client = OpenAI(
    api_key="torch-elskenrgvoiserngviopsejrmoief",
    base_url="http://127.0.0.1:6006/v1",
)

path1 = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/image.png"
path2 = "/mnt/data/llch/RAG-Challenge-2/no_git_oic/test_J.png"

model_name = "Qwen2.5-VL-32B-Instruct"
messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image_url",
            #     "image_url": {"url": image_to_base64_data_url(path1)},
            # },
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64_data_url(path2)},
            },
            {"type": "text", "text": "对图片进行OCR,输出markdown格式"},
        ],
    }
]

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.7,
)
# export no_proxy="localhost,127.0.0.1"
# uv run z_utils/llm_tools.py
print(f"{response.choices[0].message.content}")
"""
nohup python -m vllm.entrypoints.openai.api_server \
        --model /mnt/data/Qwen/Qwen2.5-VL-32B-Instruct \
        --served-model-name Qwen2.5-VL-32B-Instruct \
        --max-model-len 15360 \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 2 \
        --port 6006 \
        --host 0.0.0.0 \
        --api-key torch-elskenrgvoiserngviopsejrmoief \
        --gpu_memory_utilization 0.6 \
        --limit-mm-per-prompt image=3,video=3 \
        > vllm_Qwen2.5-VL-32B-Instruct.log 2>&1 &
        
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1+1=?"},
]


## Local file path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": "data:image;base64,/9j/..."},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": "file:///path/to/image1.jpg"},
            {"type": "image_url", "image_url": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                },
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]
"""
