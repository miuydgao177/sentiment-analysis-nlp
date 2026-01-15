import streamlit as st
from transformers import pipeline

# --- 1. 配置页面 ---
st.set_page_config(page_title="BERT 情感分析系统", layout="wide")

# --- 2. 核心逻辑：模型加载 (带缓存) ---
@st.cache_resource
def get_analyzer(model_name):
    """
    根据选择加载不同的 BERT 模型。
    - 中文推荐: shibing624/bert-base-chinese-sentiment
    - 英文推荐: distilbert-base-uncased-finetuned-sst-2-english
    """
    with st.spinner(f"正在初始化 {model_name} 模型，请稍候..."):
        try:
            return pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return None

# --- 3. 侧边栏：模型选择区 ---
with st.sidebar:
    st.title("⚙️ 模型配置")
    model_choice = st.selectbox(
        "选择分析引擎",
        ["BERT 中文模型 (高精度)", "BERT 英文模型 (标准)"]
    )

    # 映射模型名称
    model_map = {
        "BERT 中文模型 (高精度)": "shibing624/bert-base-chinese-sentiment",
        "BERT 英文模型 (标准)": "distilbert-base-uncased-finetuned-sst-2-english"
    }
    current_model = model_map[model_choice]

# --- 4. 主界面设计 ---
st.title("🧠 BERT 深度语义情感分析")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "请输入待分析的文本内容：",
        placeholder="例如：这家餐厅的服务态度非常好，菜品也很地道！",
        height=200
    )

    if st.button("🚀 开始深度分析", use_container_width=True):
        if user_input.strip():
            # 获取模型
            analyzer = get_analyzer(current_model)

            if analyzer:
                with st.spinner('BERT 正在理解语义...'):
                    # 执行分析
                    results = analyzer(user_input)
                    res = results[0]

                    # UI 显示结果
                    label = res['label']
                    score = res['score']

                    with col2:
                        st.subheader("分析结论")
                        # 简单的颜色逻辑
                        color = "green" if label in ["POSITIVE", "LABEL_1", "喜悦"] else "red"
                        st.markdown(f"### 情感标签: :{color}[{label}]")

                        st.write(f"**模型置信度:**")
                        st.progress(score)
                        st.info(f"准确率预测: {score:.2%}")

                        # 额外解释
                        if score < 0.6:
                            st.warning("注：置信度较低，建议结合人工判断。")
        else:
            st.warning("⚠️ 请先输入一些文字再点击分析。")

# --- 5. 页脚 ---
st.markdown("---")
st.caption("注：首次运行将从 Hugging Face 下载模型权重，可能需要 1-2 分钟。")