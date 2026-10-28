# arxiv-filter
Get daily new arXiv papers and filter them to the ones of interest.

The default `--backend auto` asks Llama 3.1 when CUDA is available, and otherwise uses a weighted phrase lexicon. Pass `--backend rules` to skip the model, or `--backend llm` to require it.

```bash
python llama_filter.py
python llama_filter.py --backend rules
python llama_filter.py --backend llm
```
