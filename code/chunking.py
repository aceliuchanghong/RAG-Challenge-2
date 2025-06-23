import tiktoken
from typing import List, Optional


class CustomRecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        model_name: str = "gpt-4o",
        separators: Optional[List[str]] = None,
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

    def _get_token_count(self, text: str) -> int:
        """计算字符串的 token 数量。"""
        return len(self._tokenizer.encode(text))

    def split_text(self, file_path: Optional[str] = None) -> dict:
        """
        将输入文本分割为指定格式的字典。

        参数:
            text: 要分割的文本。
            file_path: 文件路径，用于生成哈希值（可选）。

        返回:
            字典，格式为:
            {
                "file_hash": 文件哈希值,
                "chunks": [
                    {"token_count": token数量, "page_num": 页面编号, "content": 分片内容},
                    ...
                ]
            }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text:
            return {"file_hash": "", "chunks": []}

        # 1. 递归分割文本为小片段
        initial_parts = self._split(text, self._separators)

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
        # 由于输入名字都是md_{hash}.md,从这儿读取hash
        if file_path:
            import os

            # 获取文件名，例如 md_214bccfe87db220b5accebf6aae7ebba.md
            file_name = os.path.basename(file_path)
            if file_name.startswith("md_") and file_name.endswith(".md"):
                file_hash = file_name[3:-3]  # 提取 md_ 和 .md 之间的部分
            else:
                raise ValueError(
                    f"文件名格式不正确，预期为 md_{{hash}}.md，实际为 {file_name}"
                )
        else:
            raise ValueError(f"文件不存在 {file_name}")

        return {"file_hash": file_hash, "chunks": result}

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
        chunks = []
        current_chunk_parts: List[str] = []
        current_length = 0

        for part in parts:
            part_length = self._get_token_count(part)

            if current_length + part_length > self.chunk_size and current_chunk_parts:
                chunk_text = "".join(current_chunk_parts)
                chunks.append(chunk_text)

                # 处理重叠逻辑
                while (
                    current_length + part_length > self.chunk_size - self.chunk_overlap
                    and current_chunk_parts
                ):
                    first_part = current_chunk_parts.pop(0)
                    current_length -= self._get_token_count(first_part)

            current_chunk_parts.append(part)
            current_length += part_length

        if current_chunk_parts:
            chunk_text = "".join(current_chunk_parts)
            chunks.append(chunk_text)

        return chunks


if __name__ == "__main__":
    """
    uv run code/chunking.py
    """
    text_splitter = CustomRecursiveCharacterTextSplitter()
    md_file = "output/md/md_214bccfe87db220b5accebf6aae7ebba.md"

    result = text_splitter.split_text(file_path=md_file)
    print(result)
