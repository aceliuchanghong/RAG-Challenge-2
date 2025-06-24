from pathlib import Path
from typing import Union, Dict, List, Any, Tuple
from dataclasses import dataclass
import cohere
from openai import OpenAI
import json
from lancedb.pydantic import LanceModel
from tenacity import retry, wait_fixed, stop_after_attempt

import os

import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)


from code.ingest_with_lancedb import LanceDBIngestor
from z_utils.get_json import parse_and_check_json_markdown


@dataclass
class QuestionsProcessor:
    lancedb_dir: Union[str, Path] = "./lancedb"
    table_name: str = "file_chunks"
    rerank_sample_size: int = 10
    retrieve_sample_size: int = 20
    answering_model: str = "Qwen3"
    retrieve_model: str = "Qwen3-Embedding-4B"
    rerank_model: str = "mxbai_rerank_large_v2"

    def __post_init__(self):
        """初始化数据库连接和重排器。"""
        self.db = LanceDBIngestor(self.lancedb_dir)
        self.db._open_or_create_table(self.table_name)
        self.ranker = cohere.Client(
            base_url=os.getenv("RERANK_BASE_URL"),
            api_key=os.getenv("RERANK_API_KEY"),
        )
        self.llm = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    def rerank_documents(
        self, query: str, documents: List[Union[str, Dict]]
    ) -> List[Dict]:
        """
        使用连接到 vLLM 的 reranker 服务对文档进行重排序。
        """
        if not hasattr(self, "ranker"):
            raise RuntimeError("Reranker 未被正确初始化。")

        reranked_results = self.ranker.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=self.rerank_sample_size,
        )

        return [result.dict() for result in reranked_results.results]

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
    def rewrite_query(self, query: str) -> str:
        """
        使用 LLM 将用户问题重写为更适合检索的查询。
        例如，将口语化的问题转换为关键词明确的查询。
        """
        messages = [
            {
                "role": "user",
                "content": "将用户提出的口语化的问题改写成一个关键词明确的查询。"
                + "\n1. 改写后的查询应清晰、符合上下文，并通过扩展细节或重新表达来增强多样性及潜在的回答检索质量。"
                + "\n2. 改写查询应确保没有改变核心语义。"
                + "\n3. 请输出json格式数据 包含一个键 'new_query'，其值为改写后的查询。"
                + "\nExample_input: '如何优化大型语言模型的推理速度？'"
                + "\nExample_output: {'new_query': '如何从模型架构和部署方面提升大型语言模型的推理速度？'}"
                + "\n\nuser_input_query: "
                + query,
            },
        ]
        try:
            response = self.llm.chat.completions.create(
                model=self.answering_model,
                messages=messages,
                temperature=0.7,
            )
            response_content = response.choices[0].message.content
            parsed_json = parse_and_check_json_markdown(response_content, ["new_query"])
            new_query = parsed_json["new_query"]
            return new_query
        except Exception as e:
            print(f"重写查询时出错: {e}")
            return query

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
    def decompose_question(self, query: str) -> Tuple[List[str], List[str]]:
        """
        将复杂问题分解为多个独立的子问题，并进行输出校验。
        """
        messages = [
            {
                "role": "user",
                "content": "请将用户提出的复杂问题分层次地进行分解："
                + "\n1. 首先输出 high_level_sub_questions 即最核心的1~2个高层/宏观级子问题"
                + "\n2. 然后再识别原问题输出3~7个 low_level_sub_questions, 这些是具体、独立、可检索的小问题。注意是[具体]、[独立]、[可检索]的小问题"
                + "\n3. 输出格式必须为JSON 包含两个键:'high_level_sub_questions' 和 'low_level_sub_questions'，值均为字符串列表。"
                + "\n4. 如果问题本身足够简单，则 high_level_sub_questions 和 low_level_sub_questions 均只包含原始问题。"
                + "\n\n示例输入: 'vLLM和TensorRT-LLM有什么区别?'"
                + """\n示例输出: 
        ```json
        {
          "high_level_sub_questions": [
            "在大语言模型推理场景中 vLLM 和 TensorRT-LLM 各自适用于哪些典型应用场景?",
            "vLLM 和 TensorRT-LLM 在设计目标和技术实现上有何核心差异",
          ],
          "low_level_sub_questions": [
            "vLLM 是基于什么框架或技术开发的?",
            "TensorRT-LLM 是基于什么框架或技术开发的?",
            "vLLM 的推理加速原理是什么?",
            "vLLM 和 TensorRT-LLM 在部署难易程度上有何差异?",
            ...
          ]
        }
        ```
        """
                + "\n\n用户输入的问题: "
                + query,
            }
        ]
        try:
            response = self.llm.chat.completions.create(
                model=self.answering_model,
                messages=messages,
                temperature=0.7,
            )
            response_content = response.choices[0].message.content
            # 使用校验函数
            parsed_json = parse_and_check_json_markdown(
                response_content,
                ["high_level_sub_questions", "low_level_sub_questions"],
            )
            high_level_sub_questions = parsed_json["high_level_sub_questions"]
            low_level_sub_questions = parsed_json["low_level_sub_questions"]
            if not isinstance(high_level_sub_questions, list):
                raise ValueError("JSON键'high_level_sub_questions'的值不是一个列表。")
            if not isinstance(low_level_sub_questions, list):
                raise ValueError("JSON键'low_level_sub_questions'的值不是一个列表。")

            return high_level_sub_questions, low_level_sub_questions
        except Exception as e:
            print(f"分解问题时出错或校验失败: {e}，将不对问题进行分解。")
            return [response_content], []

    def _merge_unique_docs(self, vector_results, keyword_results=[]):
        all_docs = {}
        for doc in vector_results + keyword_results:
            key = (doc.report_sha1, doc.chunk_id)
            if key not in all_docs:
                all_docs[key] = doc
        return list(all_docs.values())

    def retrieve_question(self, query: str) -> List[LanceModel]:
        """
        为一个查询执行混合检索（向量+关键词）
        """
        try:
            vector_results = self.db.vector_search(
                query, limit=self.retrieve_sample_size, do_print=False
            )
        except Exception as e:
            print(f"向量检索时出错: {e}")
            vector_results = []

        try:
            keyword_results = self.db.keyword_search(
                query, limit=self.retrieve_sample_size, do_print=False
            )
        except Exception as e:
            print(f"关键词检索时出错: {e}")
            keyword_results = []
        """
        [LanceDBSchema(text='却说庞统迤逦前进及', vector=FixedSizeList(dim=2560), report_sha1='c6f5b8c6fc281b49f3b50cc778c5cecc', chunk_id=3305),
         LanceDBSchema(text='庞统字士元巴西人也', vector=FixedSizeList(dim=2560), report_sha1='c6f5b8c6fc281b49f3b50cc778c5cecc', chunk_id=3306),
         ...]
        """
        unique_docs = self._merge_unique_docs(vector_results, keyword_results)
        # print(f"混合检索后共找到 {len(unique_docs)} 份独立文档。")
        return unique_docs

    def find_question_related_docs(self, query: str) -> Dict[str, Any]:
        """
        处理单个问题的 RAG 流程
        """
        rewritten_query = self.rewrite_query(query)
        high_level_sub_questions, low_level_sub_questions = self.decompose_question(
            rewritten_query
        )
        sub_questions = high_level_sub_questions + low_level_sub_questions
        emb_doc_list = []
        for sub_question in sub_questions:
            result = self.retrieve_question(sub_question)
            emb_doc_list = emb_doc_list + result
        emb_doc_list_unique = self._merge_unique_docs(emb_doc_list)
        print(f"总共emb块数:{len(emb_doc_list_unique)}")

        docs = []
        for i, emb_doc in enumerate(emb_doc_list_unique):
            docs.append(emb_doc.text)
        reranked_docs = self.rerank_documents(query=rewritten_query, documents=docs)
        print(f"总共rerank块数:{len(reranked_docs)}")
        results = []
        for _, reranked_doc in enumerate(reranked_docs):
            i = reranked_doc["index"]
            if (
                reranked_doc["document"]["text"] == emb_doc_list_unique[i].text
                and reranked_doc["relevance_score"] > 6.5
            ):
                result = {
                    "text": reranked_doc["document"]["text"],
                    "report_sha1": emb_doc_list_unique[i].report_sha1,
                    "chunk_id": emb_doc_list_unique[i].chunk_id,
                    "relevance_score": reranked_doc["relevance_score"],
                }
                results.append(result)

        return results


if __name__ == "__main__":
    """
    export no_proxy="localhost,127.0.0.1,121.205.3.100"

    uv run code/questions_processing.py
    """
    question_processor = QuestionsProcessor()

    query = "就三国演义小说介绍一下庞统的生平"
    question_related_docs = question_processor.find_question_related_docs(query)
    print(f"{question_related_docs}")

    """
    # 测试模型接口
    # 测试llm
    
    messages = [
        {"role": "user", "content": "1+1=?"},
    ]
    response = questionsProcessor.llm.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        temperature=0.7,
    )
    print(f"{response.choices[0].message.content}\n")

    try:
        sample_query = "如何优化大型语言模型的推理速度？"
        docs = [
            "大型语言模型的推理速度可以通过模型量化来提升 例如使用INT8或INT4精度。",
            "使用闪存注意力 Flash Attention 是加速Transformer模型推理的有效方法。",
            "对于大型模型 张量并行 Tensor Parallelism 是一种常见的分布式推理策略。",
            "进行知识蒸馏，将大模型的知识迁移到小模型上，可以获得更快的推理速度。",
            "如何给宠物猫洗澡？首先要安抚它的情绪，准备好温水和宠物专用香波。",  # 无关文档
            "优化推理速度的关键在于减少计算量和内存访问 vLLM通过PagedAttention实现了这一点。",
            "剪枝 Pruning 技术可以移除模型中不重要的权重 从而减小模型大小并加速计算。",
        ]
        # 测试rerank
        reranked_docs = questionsProcessor.rerank_documents(
            query=sample_query, documents=docs
        )
        print("--- 重排后的文档顺序 (相关性从高到低) ---")
        import json

        print(json.dumps(reranked_docs, indent=2, ensure_ascii=False))

        # 测试emb
        embeddings = questionsProcessor.db._get_embeddings(docs)
        print(f"{len(embeddings),len(embeddings[0])}")

    except (ValueError, RuntimeError) as e:
        print(f"发生错误: {e}")
    """
