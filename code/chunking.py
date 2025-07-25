import tiktoken
from typing import List, Optional
import os
import sys
import re

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from code.serialize_tables import TableSerializer


class CustomRecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4o",
        separators: Optional[List[str]] = None,
        tags: tuple[str] = None,
        ser_tab: bool = False,
    ):
        """
        初始化自定义文本分割器。

        参数:
            chunk_size: 每个分片的最大 token 数量。
            chunk_overlap: 相邻分片之间的重叠 token 数量。
            model_name: 用于 tiktoken 编码的模型名称（例如 "gpt-4o", "o200k_base"）。
            separators: 分割文本的字符串列表，按优先级排序。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"重叠大小 ({chunk_overlap}) 必须小于分片大小 ({chunk_size})。"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self._separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        self._tokenizer = tiktoken.encoding_for_model(model_name)
        self.tags = tags if tags else ()
        self.ser_tab = ser_tab
        if self.ser_tab:
            self.serializer = TableSerializer()

    def _get_token_count(self, text: str) -> int:
        """计算字符串的 token 数量。"""
        return len(self._tokenizer.encode(text))

    async def split_text(self, file_path: Optional[str] = None) -> dict:
        """
        将输入文本分割为指定格式的字典。此方法现在是异步的。

        参数:
            file_path: 文件路径，用于生成哈希值（可选）。

        返回:
            字典，格式为:
            {
                "file_hash": 文件哈希值,
                "tags": ["tag1", "tag2", ...],
                "chunks": [
                    {"token_count": token数量, "page_num": 页面编号, "content": 分片内容},
                    ...
                ]
            }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text:
            return {"file_hash": "", "tags": [], "chunks": []}

        initial_parts = []

        if self.ser_tab:
            # 序列化表格
            processed_text = ""
            last_end = 0

            # 找到所有表格及其位置
            for match in re.finditer(r"<table>.*?</table>", text, re.DOTALL):
                start, end = match.span()
                table_html = match.group(0)

                # 添加表格前方的文本
                processed_text += text[last_end:start]

                # 获取上下文
                context_before = text[:start]
                context_after = text[end:]

                # 异步序列化表格==> TODO 此处需要多并发
                serialized_blocks = await self.serializer.serialize_table(
                    table_html=table_html,
                    # context_before=context_before[-100:],
                    # context_after=context_after[:100],
                )

                # 将序列化后的内容（字典列表）转换为字符串并添加到处理后的文本中
                serialized_content = "\n\n".join(
                    [item["content"] for item in serialized_blocks]
                )
                processed_text += serialized_content

                last_end = end

            # 添加最后一个表格之后剩余的文本
            processed_text += text[last_end:]

            # 对处理后（表格已被序列化文本替换）的全文进行标准切分
            initial_parts = self._split(processed_text, self._separators)

        else:
            # 将表格和图片作为不可分割的整体
            # 定义正则表达式，用于匹配 <table>...</table> 和 <image_description>...</image_description>
            # 括号 () 使 re.split 保留分隔符（即我们的目标标签）
            pattern = r"(<table>.*?</table>|<image_description>.*?</image_description>)"

            # 切分文本，结果是 [普通文本, 标签块, 普通文本, 标签块, ...] 的交替列表
            split_text = re.split(pattern, text, flags=re.DOTALL)

            for part in split_text:
                if not part:
                    continue
                # 检查当前部分是否是我们的特殊标签块
                if part.startswith("<table>") or part.startswith("<image_description>"):
                    # 如果是，直接将其作为一个整体部分添加
                    initial_parts.append(part)
                else:
                    # 如果是普通文本，则使用现有的递归分割逻辑
                    initial_parts.extend(self._split(part, self._separators))

        # 2. 合并小片段为最终分片
        final_chunks = self._merge(initial_parts)

        # 3. 组装返回结果
        result = []
        for i, chunk in enumerate(final_chunks, 1):
            result.append(
                {
                    "token_count": self._get_token_count(chunk),
                    "page_num": i,
                    "content": chunk,
                }
            )

        # 从文件名中提取哈希值
        if file_path:
            file_name = os.path.basename(file_path)
            if file_name.startswith("md_") and file_name.endswith(".md"):
                file_hash = file_name[3:-3]
            else:
                raise ValueError(
                    f"文件名格式不正确，预期为 md_{{hash}}.md 实际为 {file_name}"
                )
        else:
            raise ValueError("必须提供文件路径 (file_path)。")

        return {"file_hash": file_hash, "tags": list(self.tags), "chunks": result}

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """
        递归地将文本分割为小片段，直到每个片段小于分片大小。

        参数:
            text: 要分割的文本。
            separators: 分割符列表。

        返回:
            分割后的文本片段列表。
        """
        final_parts = []
        separator = separators[0]
        remaining_separators = separators[1:]

        # 根据当前分隔符分割文本
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 遍历分割结果，处理过大的片段
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                part_to_check = split + separator
            else:
                part_to_check = split

            if self._get_token_count(part_to_check) < self.chunk_size:
                final_parts.append(part_to_check)
            elif remaining_separators:
                final_parts.extend(self._split(part_to_check, remaining_separators))
            else:
                final_parts.append(part_to_check)

        return final_parts

    def _merge(self, parts: List[str]) -> List[str]:
        """
        将小片段合并为最终分片，满足分片大小和重叠要求。

        参数:
            parts: 小片段列表。

        返回:
            最终分片列表。
        """
        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_len = 0

        for part in parts:
            if not part.strip():
                continue

            part_len = self._get_token_count(part)
            is_special = part.startswith("<table>") or part.startswith(
                "<image_description>"
            )

            # 如果当前 chunk 非空，并且（新 part 是特殊块 或 加入后会超长），则需要结束当前 chunk
            if current_chunk_parts and (
                is_special or current_len + part_len > self.chunk_size
            ):
                chunk_text = "".join(current_chunk_parts)
                chunks.append(chunk_text)

                # 重置，为新 chunk 做准备（这里简化处理，不使用复杂的重叠逻辑，因为特殊块本身就是强分隔符）
                # 如果需要重叠，可以在这里实现
                current_chunk_parts = []
                current_len = 0

            if is_special and part_len > self.chunk_size:
                if current_chunk_parts:
                    chunks.append("".join(current_chunk_parts))
                    current_chunk_parts = []
                    current_len = 0
                chunks.append(part)
                continue

            # 将当前 part 加入 chunk
            current_chunk_parts.append(part)
            current_len += part_len

        # 处理最后一个剩余的 chunk
        if current_chunk_parts:
            chunks.append("".join(current_chunk_parts))

        return chunks


if __name__ == "__main__":
    """
    uv run code/chunking.py
    """
    import asyncio

    ser_tab = True
    # ser_tab = False

    async def main():
        text_splitter = CustomRecursiveCharacterTextSplitter(ser_tab=ser_tab)
        md_file = "output/md/md_214bccfe87db220b5accebf6aae7ebba.md"

        result = await text_splitter.split_text(file_path=md_file)
        print(result)

    asyncio.run(main())
