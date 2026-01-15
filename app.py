import os
import streamlit as st
import joblib
from transformers import pipeline

# 环境变量：确保 BERT 下载稳定
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 缓存模型加载 ---
@st.cache_resource
def load_bert():
    # 使用你之前验证通过的 RoBERTa 模型
    return pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-chinese-sentiment")

@st.cache_resource
def load_ml_model(name):
    # 加载本地保存的传统机器学习模型
    return joblib.load(f'models/{name}.pkl')

# --- 页面配置 ---
st.set_page_config(page_title="Multi-Model Sentiment Analysis", layout="wide")
st.title("Sentiment Analysis Comparative Platform")

# 侧边栏：展示工作量与选项
st.sidebar.header("Algorithm Selection")
model_type = st.sidebar.selectbox(
    "Choose Classifier",
    ["BERT (Transformer)", "SVM (Machine Learning)", "Naive Bayes (Statistical)", "Logistic Regression (Linear)"]
)

# 输入区域
input_text = st.text_area("Input Text:", "这家餐厅的服务态度非常专业，食物也很美味。")

if st.button("Execute Analysis"):
    if not input_text:
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Analyzing with {model_type}..."):
            if model_type == "BERT (Transformer)":
                classifier = load_bert()
                res = classifier(input_text)[0]
                label, score = res['label'], res['score']
            else:
                # 传统 ML 处理逻辑
                file_name = model_type.split(' (')[0].lower().replace(' ', '_')
                model = load_ml_model(file_name)
                prediction = model.predict([input_text])[0]
                # 获取概率作为置信度
                prob = model.predict_proba([input_text])[0]
                score = prob[prediction]
                label = "Positive" if prediction == 1 else "Negative"

            # 结果呈现
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            col1.metric("Sentiment Label", label)
            col2.metric("Confidence Score", f"{score:.4f}")