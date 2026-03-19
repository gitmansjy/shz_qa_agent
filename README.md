# 水浒传AI问答小助手

这是一个使用Streamlit和Qwen模型构建的简单AI问答助手。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
streamlit run app.py
```

然后在浏览器中打开 http://localhost:8501

## 配置

在 `app.py` 中设置您的 `DASHSCOPE_API_KEY`。

**注意**：请将API密钥保存在安全的地方，避免泄露。建议将 `app.py` 加入 `.gitignore` 并使用环境变量。
