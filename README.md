# BERT Sentiment Analysis

High-precision Chinese sentiment classification system utilizing the RoBERTa-base architecture.

## Implementation
The system employs the `uer/roberta-base-finetuned-chinese-sentiment` model. It features a deferred loading strategy to optimize memory utilization and utilizes local caching via verified endpoints for stable deployment.

## Technical Stack
* Framework: Streamlit
* Model: RoBERTa-base (Chinese)
* Inference: Hugging Face Transformers

## Execution
```bash
pip install streamlit transformers torch
export HF_ENDPOINT=https://hf-mirror.com
python3 -m streamlit run bert_demo.py
```

