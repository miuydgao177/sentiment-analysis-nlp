import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. 准备目录
os.makedirs('models', exist_ok=True)

# 2. 模拟中文情感数据（用于生成模型文件）
texts = [
    "这个产品非常好用", "质量很棒", "体验极佳", "我很满意", 
    "非常差劲", "质量很烂", "体验不好", "我很失望", "客服态度很差"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 为正向, 0 为负向

# 3. 训练并保存模型函数
def train_and_save(model, name):
    # 使用 TF-IDF 向量化 + 模型 构成 Pipeline
    clf = make_pipeline(TfidfVectorizer(), model)
    clf.fit(texts, labels)
    joblib.dump(clf, f'models/{name}.pkl')
    print(f"Model saved: models/{name}.pkl")

# 4. 执行训练任务
train_and_save(MultinomialNB(), "naive_bayes")
train_and_save(SVC(probability=True), "svm")
train_and_save(LogisticRegression(), "logistic_regression")