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
