# RAG-Challenge-2

## Usage

You can run any part of pipeline by uncommenting the method you want to run in `src/pipeline.py` and executing:
```bash
python .\src\pipeline.py
```

You can also run any pipeline stage using `main.py`, but you need to run it from the directory containing your data:
```bash
cd .\data\test_set\
python ..\..\main.py process-questions --config max_nst_o3m
```

### CLI Commands

Get help on available commands:
```bash
python main.py --help
```

Available commands:
- `download-models` - Download required docling models
- `parse-pdfs` - Parse PDF reports with parallel processing options
- `serialize-tables` - Process tables in parsed reports
- `process-reports` - Run the full pipeline on parsed reports
- `process-questions` - Process questions using specified config

Each command has its own options. For example:
```bash
python main.py parse-pdfs --help
# Shows options like --parallel/--sequential, --chunk-size, --max-workers

python main.py process-reports --config ser_tab
# Process reports with serialized tables config
```

## Some configs

- `max_nst_o3m` - Best performing config using OpenAI's o3-mini model
- `ibm_llama70b` - Alternative using IBM's Llama 70B model
- `gemini_thinking` - Full context answering with using enormous context window of Gemini. It is not RAG, actually

Check `pipeline.py` for more configs and detils on them.


### 通用思路
```
[用户问题]
     ↓
[Query Preprocessor] → 清洗、意图识别、是否存在判断（Existence Checker）
     ↓
[Router] → 决定是否需跨文档检索或多跳检索
     ↓
[Retriever] → 多路召回 Top-k 文档（BM25 + Dense Retriever）
     ↓
[Reranker] → 使用 LLM 或 RankLLM 进行语义相关性排序
     ↓
[Context Assembler] → 构造高质量上下文（含页码、来源标注）
     ↓
[Generator] → 使用高精度 LLM 生成答案
     ↓
[Postprocessor] → 格式校验（Schema Validator）、引用标注、N/A 判断
     ↓
[Answer Output] → 返回结构化 JSON 答案
```



1. **下载模型**  
   - 运行 `python main.py download_models`，下载 PDF 解析（如 Docling）所需的模型，确保后续步骤顺利进行。

2. **解析 PDF 报告**  
   - 运行 `python main.py parse_pdfs --parallel --chunk-size 2 --max-workers 10`，以并行方式解析 PDF 报告。  
   - 此步骤使用 `src/pdf_parsing/PDFParser.py` 中的 `PDFParser` 类，基于 Docling 提取文本和结构，输出存储在 `01_parsed_reports` 目录。  
   - 根据文章描述，解析 100 个 PDF（每份最多 1000 页）耗时 40 分钟，使用 GPU（如 4090）可加速。

3. **序列化表格（可选）**  
   - 运行 `python main.py serialize_tables --max-workers 10`，处理解析后的报告中的表格。  
   - 此步骤使用 `src/tables_serialization/TableSerializer.py`，但文章提到最终解决方案未使用序列化表格，因其略微降低效果，可根据需要选择是否执行。

4. **处理报告**  
   - 运行 `python main.py process_reports --config no_ser_tab`，处理解析后的报告。  
   - 此步骤包括多个子阶段：  
     - **合并报告**：使用 `merge_reports` 方法，调用 `src/parsed_reports_merging/PageTextPreparation.py` 合并多页数据。  
     - **导出为 Markdown**：使用 `export_reports_to_markdown` 将合并后的报告转换为 Markdown 格式，便于后续处理。  
     - **文本分割**：使用 `chunk_reports`，调用 `src/text_splitter/TextSplitter.py`，将文本分割为 300 令牌的块，50 令牌重叠。  
     - **创建向量数据库**：使用 `create_vector_dbs`，调用 `src/ingestion/VectorDBIngestor.py`，使用 FAISS 和 text-embedding-3-large 嵌入创建向量数据库。  
   - 配置 `no_ser_tab` 表示不使用序列化表格，适合最佳性能。

5. **处理问题**  
   - 运行 `python main.py process_questions --config max_nst_o3m`，处理问题以生成答案。  
   - 此步骤使用 `src/questions_processing/QuestionsProcessor.py`，包括：  
     - 检索：从向量数据库中获取前 30 个块。  
     - 重新排序：使用 LLM（如 GPT-4o-mini）重新排序，成本小于 1 美分/问题，最终选择前 10 页。  
     - 生成：使用链式思维（Chain of Thought）、结构化输出和单次提示生成答案，支持比较查询通过多查询路由处理。  
   - 系统可在 2 分钟内完成 100 个问题，最初目标为 10 分钟限制（后延长）。

#### 代码模块学习
为了深入理解，逐个研究以下关键 Python 文件，理解其功能：

| **模块**                          | **描述**                                      | **文件路径**                          |
|-----------------------------------|----------------------------------------------|---------------------------------------|
| PDF 解析                          | 使用 Docling 解析 PDF，提取文本和结构         | `src/pdf_parsing/PDFParser.py`        |
| 表格序列化                        | 处理报告中的表格（可选）                      | `src/tables_serialization/TableSerializer.py` |
| 报告合并                          | 合并解析后的多页数据                          | `src/parsed_reports_merging/PageTextPreparation.py` |
| 文本分割                          | 将文本分割为块，300 令牌，50 令牌重叠         | `src/text_splitter/TextSplitter.py`   |
| 向量数据库创建                    | 使用 FAISS 和嵌入创建向量数据库               | `src/ingestion/VectorDBIngestor.py`   |
| BM25 数据库创建（若使用）          | 创建基于 BM25 的数据库                        | `src/ingestion/BM25Ingestor.py`       |
| 问题处理                          | 检索、重新排序和生成答案                      | `src/questions_processing/QuestionsProcessor.py` |


教程：
- [How To Build an AI Knowledge Base With RAG](https://dzone.com/articles/how-to-build-an-ai-knowledge-base-with-rag)：介绍如何构建 AI 知识库。
- [Building a RAG Application from Scratch: A Beginner's Guide](https://www.pingcap.com/article/building-a-rag-application-from-scratch-a-beginners-guide/)：从零开始构建 RAG 应用的指南。
- [Building a Knowledge Base for RAG: A Step-by-Step Guide](https://medium.com/%40arushiagg04/building-a-knowledge-base-for-rag-a-step-by-step-guide-c3afbccf3700)：详细步骤指南。



### rag通用难点
|难点|具体体现|
|-------|--------|
|海量异构 PDF 解析-文档多样性与解析难度|包含 PDF、Word、网页等多种格式；存在双栏、旋转表格、图表混排等结构，导致通用解析器效果不佳|
|检索准确性要求高-严格 JSON & 引用格式|检索内容需高度相关，否则生成结果会偏离问题核心，影响整体性能|
|语义匹配复杂度|用户提问方式多样，需理解同义词、上下文依赖、意图模糊等问题才能精准检索|
|上下文长度限制|大多数模型有输入长度限制，如何有效截断/压缩/合并检索到的内容成为挑战|
|生成质量控制|即使检索准确，生成过程也可能引入幻觉、冗余或格式错误，影响最终输出质量|
|数据缺失与噪声干扰|存在信息不全、错误数据、无意义问题等情况，需具备判断是否存在答案的能力|
|实时性与效率要求|在限定时间内完成从检索到生成的全流程，对系统吞吐量和响应延迟提出较高要求|
|格式与结构化输出约束|输出需严格符合 JSON Schema 或其他指定格式，容错率低，需额外后处理机制|
|资源消耗与成本控制|多次调用大模型、高频检索带来高昂的 GPU 和 API 成本，需权衡精度与开销|
|可解释性与引用溯源|要求生成内容附带来源页码或文档片段，增强可信度，但增加了系统设计复杂度|
|多文档对比与整合|部分问题涉及跨文档比较，需设计专门的路由逻辑或多阶段检索策略|
|模型偏差与公平性|不同模型可能偏向特定类型内容或表达方式，影响结果一致性与客观性|
|流水线稳定性|各模块之间耦合性强，一处出错可能导致整个流程失败，需良好的异常处理机制|
|评估指标透明化|评分标准公开且严格，系统必须保证每一步骤都可复现、可验证，避免“黑盒”操作|

### Reference
- [教程](https://gemini.google.com/app/c319b9cc7507faa0)
- [代码地址](https://github.com/aceliuchanghong/RAG-Challenge-2)
- 
