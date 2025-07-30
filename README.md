## NEW-RAG

### 环境安装

```shell
uv run install.py
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 使用说明

- **使用帮助**

```bash
uv run run.py --help
uv run run.py read-files --help
uv run run.py chunk-markdown --help
uv run run.py save-jsonl --help
...
```

- 单个命令测试示例

```shell
uv run run.py read-files --file-path no_git_oic/test_files/流式细胞制备方案.pdf
uv run run.py read-files --file-path no_git_oic/test_files/

uv run run.py chunk-markdown --md-file-path output/md/md_2a756c2048842968844b3d504cfd33b0.md
uv run run.py chunk-markdown --md-file-path output/md/

uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_2a756c2048842968844b3d504cfd33b0.jsonl
uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md
uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md --table-name new_test

uv run run.py answer-question --question "流式细胞制备如何操作?" --stream
uv run run.py answer-question --question "就三国演义小说介绍一下庞统的生平" --stream --table-name sanguo --complicated-question

uv run run.py read-files --file-path no_git_oic/test_files/三国演义.docx
uv run run.py chunk-markdown --md-file-path output/md/md_c6f5b8c6fc281b49f3b50cc778c5cecc.md
uv run run.py save-jsonl --jsonl-path-or-dir output/chunked_md/chunked_c6f5b8c6fc281b49f3b50cc778c5cecc.jsonl --table-name sanguo
```

- api服务启动

```shell
# 具体可看`api_server.py`文件里面下方 readme
uvicorn api_server:app --host 0.0.0.0 --port 5000
```

### 思路
```
[用户问题]
     ↓
[Query Preprocessor] → 清洗、意图识别、是否存在判断（Existence Checker）
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

### 模型使用为何

```
问题回答接口:
     question-->[llm]重写问题-->[emb]检索答案-->[rerank]重排-->[llm]生成答案
文档检索接口:
     question-->[llm]重写问题-->[emb]检索答案-->[rerank]重排
可读文档保存接口:
     files-->标准md-->切分-->[emb]向量化-->[lancedb]保存到指定表里面
不可读pdf/图片类文档保存接口:
     files-->[minerU]还原md-->[llm]处理表格-->标准md-->基于表格图片的切分-->[emb]向量化-->[lancedb]保存到指定表里面
                         ↘                 ↗
                           ↘             ↗
                             [vlm]处理图片
```

### 难点

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
