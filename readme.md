# RAG-TQA 项目

本项目实现了一个基于检索增强生成（RAG）的问答系统，使用 TriviaQA 数据集和 OpenAI API。

## 功能

- 加载本地 TriviaQA 数据集
- 使用 TF-IDF 进行文档检索
- 调用 OpenAI API 生成基于检索结果的答案

## 环境要求

- Python 3.7+
- Windows 操作系统

## 安装依赖

1. 克隆本仓库或下载代码到本地。
2. 打开终端，导航到项目目录。
3. 创建并激活虚拟环境（可选）：

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. 安装所需的依赖包：

   ```bash
   pip install -r requirements.txt
   ```

## 配置 OpenAI API 密钥

为了安全起见，请使用环境变量来管理 OpenAI API 密钥。

1. 设置环境变量 `OPENAI_API_KEY`：

   ```bash
   set OPENAI_API_KEY=your_openai_api_key
   ```

   你可以在系统环境变量中永久设置，或每次运行前设置。

2. 修改 `RAG.py` 中的 API 调用部分，使用环境变量：

   ```python
   import os
   import openai

   openai.api_key = os.getenv("OPENAI_API_KEY")
   ```

## 使用方法

1. 确保 TriviaQA 数据集文件位于 `TriviaQA/` 目录下。
2. 运行主程序：

   ```bash
   python RAG.py
   ```

3. 按提示输入问题，系统将输出生成的答案。

## 示例

```bash
请输入您的问题 (或输入'退出'结束): 将光速转换为千米每秒是多少？
回答: 光速约为299,792公里每秒。
```

## 注意事项

- 确保 OpenAI API 密钥有效且有足够的调用额度。
- 数据集文件路径在 RAG.py 中已预设，如有变动请相应修改。


### 部署步骤总结

1. **克隆仓库**：将项目代码下载到本地。
2. **安装依赖**：根据 `requirements.txt` 安装所需的 Python 包。
3. **配置 API 密钥**：设置 `OPENAI_API_KEY` 环境变量。
4. **准备数据集**：确保 TriviaQA 数据集文件位于指定路径。
5. **运行程序**：执行 RAG.py并按提示操作。
