# Multi-Model Sentiment Analysis Platform

Comparative sentiment analysis platform supporting Transformer-based deep learning and traditional statistical methods.

## Features
- **Multi-Model Support**: Implements four distinct algorithms (RoBERTa, SVM, Naive Bayes, Logistic Regression).
- **Comparative Analysis**: Provides a benchmark interface to evaluate Transformer-based deep learning vs. traditional statistical methods.
- **Infrastructure**: Local caching and model serialization (.pkl) are used to ensure deployment efficiency.

## Technical Stack
* Framework: Streamlit
* Models: RoBERTa-base (Chinese), SVM, Naive Bayes, Logistic Regression
* Inference: Hugging Face Transformers, Scikit-learn

## Execution
```bash
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
python3 -m streamlit run app.py
```

