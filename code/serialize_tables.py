import asyncio
import os
import json
from typing import List
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from z_utils.get_json import parse_json_markdown
from z_utils.sqlite_cache import cache_to_sqlite_async


class TableSerializer:
    def __init__(self, model: str = "Qwen3"):
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("请设置 API_KEY 环境变量")
        base_url = os.getenv("BASE_URL")
        if not base_url:
            raise ValueError("请设置 BASE_URL 环境变量")
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _build_user_prompt(
        self, table_html: str, context_before: str, context_after: str
    ) -> str:
        instructions = (
            "你是一个专业的表格序列化助手。\n"
            "你的任务是基于给定的HTML表格和它周围的文本，创建一组上下文完全独立的信息块。\n"
            "这些信息块必须是完全自包含的，因为它们将被用作独立的文档块存入向量数据库中，用于后续的检索问答。"
        )
        format_requirements = (
            "请严格遵循以下JSON格式输出。你的整个回答必须是一个单独的、格式正确的JSON数组，数组中的每个对象代表一个信息块，且只包含一个 `content` 键。\n"
            "确保 `content` 的值是完整的、自包含的句子，包含了所有必要的上下文信息（如表头、单位、标题等），以便独立理解。\n"
            "JSON格式示例:\n"
            "```json\n"
            "[\n"
            '  {"content": "A产品在2023年第一季度的销售额为150万元。"},\n'
            '  {"content": "B产品在2023年第二季度的销售额为180万元。"}\n'
            "]\n"
            "```\n"
            "不要在你的回答中包含除了这个JSON数组之外的任何额外文本、解释或注释"
        )
        input_data = []
        if context_before:
            input_data.append(f"--- 表格前置上下文如下 ---\n{context_before}\n")
        input_data.append(f"--- 需要转化的HTML表格如下 ---\n{table_html}\n")
        if context_after:
            input_data.append(f"--- 表格后置上下文 ---\n{context_after}\n")
        final_prompt = (
            f"{instructions}\n\n"
            f"{format_requirements}\n\n"
            f"## 请处理以下输入数据:\n\n"
            f"{''.join(input_data)}\n"
            "现在，请开始生成序列化的JSON信息块。"
        )
        return final_prompt

    @cache_to_sqlite_async(debug=False)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),  # 在任何异常时重试
        before_sleep=lambda retry_state: print(
            f"重试 {retry_state.attempt_number} 次，因错误: {retry_state.outcome.exception()}"
        ),
    )
    async def serialize_table(
        self, table_html: str, context_before: str = "", context_after: str = ""
    ) -> List[str]:
        """
        delete from stock_cache where params='3a5258add63de7216e6acfc68a2c79d2'

        返回示例:
        [
            {'content': '根据公司年度报告，A产品在2023年第一季度的销售额为150万元，第二季度的销售额为200万元，总计销售额为350万元。'},
            {'content': '根据公司年度报告，B产品在2023年第一季度的销售额为120万元，第二季度的销售额为180万元，总计销售额为300万元。'}
        ]
        """
        user_prompt = self._build_user_prompt(table_html, context_before, context_after)
        print(f"正在向模型 '{self.model}' 发送表格序列化请求 (要求JSON输出)...")
        try:
            if len(table_html) > 12800:
                print("警告：原始表格过长,怀疑错误输出。")
                return [{"content": table_html}]
            if "<table><tr><td>旧底图总号</td><td>" in table_html:
                print("警告：表格包含旧底图总号，多半没什么用")
                return [{"content": table_html}]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                # response_format={"type": "json_object"},
            )
            json_output_string = response.choices[0].message.content
            parsed_blocks = parse_json_markdown(json_output_string)
            if parsed_blocks:
                print(f"序列化成功！从表格中提取了 {len(parsed_blocks)} 个信息块。")
                return parsed_blocks
            else:
                print(
                    "警告：模型返回的内容无法解析或为空。将返回原始表格作为降级方案。"
                )
                return [{"content": table_html}]
        except Exception as e:
            print(f"调用大模型时发生错误: {e}")
            raise


async def main_test():
    sample_table = """
    <table>
      <caption>2023年产品销售额（单位：万元）</caption>
      <tr>
        <th>产品</th>
        <th>第一季度</th>
        <th>第二季度</th>
        <th>总计</th>
      </tr>
      <tr>
        <td>A产品</td>
        <td>150</td>
        <td>200</td>
        <td>350</td>
      </tr>
      <tr>
        <td>B产品</td>
        <td>120</td>
        <td>180</td>
        <td>300</td>
      </tr>
    </table>
    """
    sample_context_before = "根据公司年度报告，以下是主要产品的销售表现。"
    serializer = TableSerializer()
    serialized_blocks = await serializer.serialize_table(
        table_html=sample_table, context_before=sample_context_before
    )
    print("\n--- 序列化结果 ---")
    if len(serialized_blocks) == 1 and serialized_blocks[0] == sample_table:
        print("序列化失败，返回原始HTML。")
        print(serialized_blocks[0])
    else:
        for i, block in enumerate(serialized_blocks):
            print(f"--- 信息块 {i+1} ---\n{block}\n")
    print(f"{serialized_blocks}")


if __name__ == "__main__":
    # uv run code/serialize_tables.py
    asyncio.run(main_test())
