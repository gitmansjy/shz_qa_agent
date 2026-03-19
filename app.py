import streamlit as st
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional
import hashlib
import warnings
import re

# LangChain 相关导入
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.language_models.llms import LLM

# 阿里云通义千问
import dashscope
from dashscope import Generation

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 在这里设置你的API Key
DASHSCOPE_API_KEY = "替换为你的API Key"  # 替换为你的API Key
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
dashscope.api_key = DASHSCOPE_API_KEY

# 支持的文件格式
SUPPORTED_EXTENSIONS = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.doc': Docx2txtLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.xls': UnstructuredExcelLoader,
    '.csv': CSVLoader,
}

# 需要过滤的版权声明关键词
COPYRIGHT_KEYWORDS = [
    "爱上阅读", "www.isyd.net", "isyd.net",
    "声明：本书来自互联网",
    "敬告：请在下载后的24小时内删除",
    "书名：", "作者：",
    "章节数：", "字数：",
    "========简介========",
    "本书由", "搜集整理",
    "版权归作者所有",
    "仅供参考", "查阅资料"
]

# 正文开始标记
CONTENT_START_MARKERS = [
    "话说", "第一回", "第二回", "第三回", "第四回", "第五回",
    "引首", "楔子", "诗曰", "词曰",
    "且说", "却说", "只见", "当时",
    "第一章", "第二章", "第三章",
]

# 宋江之死相关关键词（用于提高检索权重）
DEATH_KEYWORDS = [
    "宋江", "死", "鸩", "毒", "酒", "葬", "墓", "庙", 
    "封", "蓼儿洼", "楚州", "黄壤", "显灵", "招安",
    "第一百二十回", "第120回", "最后一回"
]

# ==================== 自定义LLM类 ====================
class DashScopeLLM(LLM):
    """自定义 DashScope LLM 包装器"""
    
    model: str = "qwen-plus"
    temperature: float = 0.2
    api_key: str = DASHSCOPE_API_KEY
    
    @property
    def _llm_type(self) -> str:
        return "dashscope"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """调用 DashScope API"""
        try:
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                api_key=self.api_key
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"API调用失败: {response.message}"
        except Exception as e:
            return f"错误: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}


# ==================== 文档清理函数 ====================
def clean_document_content(content: str, filename: str = "") -> str:
    """
    清理文档内容，移除版权声明等无关信息
    """
    original_length = len(content)
    
    if len(content) < 500:
        return content
    
    # 尝试找到正文开始的位置
    content_start = 0
    lines = content.split('\n')
    for i, line in enumerate(lines[:50]):
        for marker in CONTENT_START_MARKERS:
            if marker in line:
                content_start = sum(len(l) + 1 for l in lines[:i])
                print(f"  在文件 {filename} 的第 {i+1} 行找到正文标记: '{marker}'")
                break
        if content_start > 0:
            break
    
    # 如果没有找到标记，尝试根据版权关键词判断
    if content_start == 0:
        preview = content[:1000]
        copyright_count = sum(1 for keyword in COPYRIGHT_KEYWORDS if keyword in preview)
        if copyright_count >= 3:
            content_start = int(len(content) * 0.3)
            print(f"  在文件 {filename} 中发现 {copyright_count} 个版权关键词，跳过前30%")
    
    cleaned = content[content_start:] if content_start > 0 else content
    
    # 移除剩余的单行版权声明
    cleaned_lines = []
    for line in cleaned.split('\n'):
        if not any(keyword in line for keyword in COPYRIGHT_KEYWORDS):
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    print(f"  清理前: {original_length} 字符, 清理后: {len(cleaned)} 字符")
    
    return cleaned


# ==================== 文档加载函数 ====================
def load_documents_from_folder(folder_path):
    """
    从文件夹加载所有支持的文档
    """
    folder = Path(folder_path)
    if not folder.exists():
        return [], [], {"error": "文件夹不存在"}
    
    all_documents = []
    failed_files = []
    file_stats = {
        'total_files': 0,
        'supported_files': 0,
        'unsupported_files': 0,
        'by_type': {},
        'file_details': []
    }
    
    print(f"\n📂 扫描文件夹: {folder_path}")
    
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            file_stats['total_files'] += 1
            file_extension = file_path.suffix.lower()
            
            print(f"发现文件: {file_path.name} (类型: {file_extension})")
            
            if file_extension in SUPPORTED_EXTENSIONS:
                file_stats['supported_files'] += 1
                file_stats['by_type'][file_extension] = file_stats['by_type'].get(file_extension, 0) + 1
                
                try:
                    documents = []
                    
                    if file_extension == '.txt':
                        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
                        loaded = False
                        
                        for encoding in encodings:
                            try:
                                loader = SUPPORTED_EXTENSIONS[file_extension](str(file_path), encoding=encoding)
                                documents = loader.load()
                                loaded = True
                                print(f"  ✅ 使用 {encoding} 编码加载成功")
                                break
                            except Exception as e:
                                print(f"  ⚠️  {encoding} 编码失败: {e}")
                                continue
                        
                        if not loaded:
                            print(f"  ❌ 所有编码都失败，无法加载文件")
                            failed_files.append(f"{file_path.name}: 编码问题")
                            continue
                    else:
                        loader = SUPPORTED_EXTENSIONS[file_extension](str(file_path))
                        documents = loader.load()
                        print(f"  ✅ 加载成功")
                    
                    if documents and len(documents) > 0:
                        original_content = documents[0].page_content
                        cleaned_content = clean_document_content(original_content, file_path.name)
                        
                        documents[0].page_content = cleaned_content
                        
                        file_stats['file_details'].append({
                            'name': file_path.name,
                            'encoding': encoding if file_extension == '.txt' else 'binary',
                            'original_size': len(original_content),
                            'cleaned_size': len(cleaned_content),
                            'preview': cleaned_content[:100]
                        })
                        
                        for doc in documents:
                            doc.metadata['source'] = str(file_path)
                            doc.metadata['file_name'] = file_path.name
                            doc.metadata['file_type'] = file_extension
                            doc.metadata['file_path'] = str(file_path.relative_to(folder))
                            doc.metadata['cleaned'] = True
                        
                        all_documents.extend(documents)
                    else:
                        print(f"  ⚠️  文档内容为空")
                        failed_files.append(f"{file_path.name}: 内容为空")
                    
                except Exception as e:
                    print(f"  ❌ 加载失败: {e}")
                    failed_files.append(f"{file_path.name}: {str(e)}")
            else:
                file_stats['unsupported_files'] += 1
                print(f"  ⚠️  不支持的格式")
    
    print(f"\n📊 加载统计:")
    print(f"  总文件数: {file_stats['total_files']}")
    print(f"  成功加载: {file_stats['supported_files']}")
    print(f"  文档片段数: {len(all_documents)}")
    
    return all_documents, failed_files, file_stats


# ==================== 检索增强函数 ====================
def filter_copyright_docs(docs):
    """过滤掉包含版权声明的文档"""
    filtered = []
    for doc in docs:
        content = doc.page_content
        if any(keyword in content for keyword in COPYRIGHT_KEYWORDS):
            continue
        if len(content) < 50:
            continue
        filtered.append(doc)
    return filtered


def rewrite_question_with_llm(question):
    """
    使用大模型理解并改写问题，使其更适合在《水浒传》中检索
    
    Args:
        question: 用户原始问题
        
    Returns:
        改写后的查询语句
    """
    prompt = f"""你是一个专业的问题改写助手，专门为《水浒传》知识问答系统服务。
请将用户的问题改写成更适合在《水浒传》原文中检索的形式。

用户问题：{question}

改写要求：
1. 提取核心关键词，包括同义词和近义词
2. 如果是问人物结局，要加上"死"、"亡"、"结局"、"最后"等词
3. 保留具体人名、地名、事件名
4. 如果是问"怎么死的"，要特别加入"鸩"、"毒"、"酒"等具体死亡方式的关键词
5. 对于宋江，要考虑到他的结局在第120回，可以加入"第120回"、"最后一回"等章节信息
6. 输出格式：只输出改写后的查询语句，不要任何解释和额外文字

改写结果："""
    
    try:
        response = Generation.call(
            model="qwen-plus",
            prompt=prompt,
            temperature=0.1,
            api_key=DASHSCOPE_API_KEY
        )
        
        if response.status_code == 200:
            rewritten = response.output.text.strip()
            if rewritten:
                return rewritten
    except Exception as e:
        print(f"问题改写失败: {e}")
    
    return question


def rerank_docs_by_question(question: str, docs: List) -> List:
    """
    根据问题对文档进行重排序
    """
    if not docs:
        return docs
    
    words = re.findall(r'[\u4e00-\u9fff]+|\w+', question)
    keywords = [w for w in words if len(w) > 1]
    
    is_death_question = any(kw in question for kw in ["死", "毒", "鸩", "葬", "墓"])
    
    scored_docs = []
    for doc in docs:
        content = doc.page_content
        score = 0
        
        # 基础关键词匹配
        for keyword in keywords:
            count = content.count(keyword)
            score += count * 2
        
        # 第120回特殊加分
        if "第一百二十回" in content or "第120回" in content or "蓼儿洼" in content:
            score += 10
        
        # 如果是关于"死"的问题，给相关关键词更高权重
        if is_death_question or "宋江" in question:
            for kw in DEATH_KEYWORDS:
                if kw in content:
                    weight = 8 if kw in ["鸩", "毒", "死"] else 4
                    score += weight * content.count(kw)
        
        # 惩罚版权内容
        if any(kw in content for kw in COPYRIGHT_KEYWORDS):
            score -= 10
        
        # 长度奖励
        if len(content) > 500:
            score += 2
        
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    result = [doc for score, doc in scored_docs if score > -5]
    return result if result else docs[:2]


# ==================== 知识库加载函数 ====================
@st.cache_resource(show_spinner=False)
def load_knowledge_base(folder_path):
    """
    从文件夹加载知识库（带缓存）
    """
    try:
        progress_bar = st.progress(0, text="开始加载知识库...")
        
        # 步骤1: 加载所有文档
        progress_bar.progress(10, text="正在扫描文件夹并加载文档...")
        documents, failed_files, file_stats = load_documents_from_folder(folder_path)
        
        if not documents:
            st.error("没有找到可用的文档！")
            progress_bar.empty()
            return None, None, file_stats
        
        # 步骤2: 分割文档
        progress_bar.progress(30, text=f"正在分割文档 (共{len(documents)}个文档片段)...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        
        # 过滤分割后的片段
        docs = [doc for doc in docs if len(doc.page_content) > 50 and 
                not any(kw in doc.page_content for kw in COPYRIGHT_KEYWORDS)]
        
        # 步骤3: 创建向量存储
        progress_bar.progress(60, text=f"正在创建向量数据库 ({len(docs)}个文本块)...")
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()[:8]
        persist_dir = f"./chroma_db_{folder_hash}"
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            vectorstore = Chroma(
                persist_directory=persist_dir, 
                embedding_function=embeddings
            )
            st.sidebar.info("📦 使用缓存的向量数据库")
        else:
            vectorstore = Chroma.from_documents(
                docs, 
                embeddings,
                persist_directory=persist_dir
            )
            st.sidebar.info("🆕 创建新的向量数据库")
        
        # 创建基础检索器
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 25,
                "lambda_mult": 0.6
            }
        )
        
        # 步骤4: 创建LLM
        progress_bar.progress(80, text="正在初始化问答模型...")
        llm = DashScopeLLM()
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_template(
            """你是一个专业的《水浒传》知识助手，专注于回答关于《水浒传》原著的问题。

基于以下从《水浒传》原文中检索到的内容回答问题：

{context}

问题：{question}

回答要求：
1. 只基于上面提供的原文内容回答
2. 如果原文中有相关信息，请详细回答并引用原文片段
3. 如果原文中没有找到相关信息，请说"抱歉，在《水浒传》原文中没有找到相关信息"
4. 回答要准确、详细，用中文

回答："""
        )
        
        # ===== 修复后的增强检索函数（移除了st.spinner） =====
        def enhanced_retrieve_with_llm(question):
            """
            使用大模型理解并改写问题，然后进行检索
            （修复：移除了st.spinner，避免NoSessionContext错误）
            """
            # 步骤1: 用大模型改写问题
            # 使用print在终端显示（不会引起Streamlit错误）
            print(f"🤔 正在理解问题: {question}")
            rewritten_query = rewrite_question_with_llm(question)
            
            # 将改写结果保存到session_state，稍后显示
            # 注意：这里可以操作st.session_state
            import streamlit as st
            st.session_state.last_rewritten_query = rewritten_query
            
            # 步骤2: 用改写后的查询检索
            docs1 = base_retriever.invoke(rewritten_query)
            
            # 步骤3: 同时用原始问题检索
            docs2 = base_retriever.invoke(question)
            
            # 步骤4: 专门针对宋江之死问题的特殊处理
            if "宋江" in question and any(w in question for w in ["死", "怎么", "结局", "毒"]):
                death_query = "第120回 宋江 鸩 毒 死 蓼儿洼 最后一回"
                docs3 = base_retriever.invoke(death_query)
            else:
                docs3 = []
            
            # 步骤5: 合并所有结果，去重
            all_docs = []
            seen_ids = set()
            
            for doc in docs1 + docs2 + docs3:
                if id(doc) not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(id(doc))
            
            # 步骤6: 过滤版权内容
            all_docs = filter_copyright_docs(all_docs)
            
            # 步骤7: 重排序
            all_docs = rerank_docs_by_question(question, all_docs)
            
            # 记录检索到的文档数量
            st.session_state.last_retrieved_count = len(all_docs)
            
            return all_docs
        
        # 格式化文档函数
        def format_docs_with_sources(docs):
            if not docs:
                return "没有找到相关原文内容。"
            
            formatted_docs = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get('file_name', '未知来源')
                content = doc.page_content.strip()
                formatted_docs.append(f"[来自《{source}》]\n{content}")
            
            return "\n\n---\n\n".join(formatted_docs)
        
        # 构建问答链
        chain = (
            RunnableParallel({
                "context": lambda x: format_docs_with_sources(enhanced_retrieve_with_llm(x)),
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        
        progress_bar.progress(100, text="✅ 知识库加载完成！")
        progress_bar.empty()
        
        file_stats['chunk_count'] = len(docs)
        
        return chain, enhanced_retrieve_with_llm, file_stats
        
    except Exception as e:
        st.error(f"加载知识库时出错: {e}")
        import traceback
        with st.expander("查看详细错误"):
            st.code(traceback.format_exc())
        return None, None, {}


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="水浒传知识问答助手",
    page_icon="📚",
    layout="wide"
)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_folder' not in st.session_state:
    st.session_state.current_folder = ""

if 'chain' not in st.session_state:
    st.session_state.chain = None

if 'retriever_func' not in st.session_state:
    st.session_state.retriever_func = None

if 'file_stats' not in st.session_state:
    st.session_state.file_stats = {}

# 初始化用于显示改写结果的变量
if 'last_rewritten_query' not in st.session_state:
    st.session_state.last_rewritten_query = None

if 'last_retrieved_count' not in st.session_state:
    st.session_state.last_retrieved_count = 0


# ==================== 侧边栏界面 ====================
with st.sidebar:
    st.title("📁 文档管理")
    
    folder_path = st.text_input(
        "输入文件夹路径",
        value=r"D:\py\study\books",
        placeholder="例如: D:/my_documents"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        load_button = st.button("📂 加载知识库", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 重置对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    if st.button("🗑️ 清除缓存并重新加载", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.session_state.chain = None
        st.session_state.retriever_func = None
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("📄 支持的文件格式", expanded=False):
        st.markdown("""
        - 📝 文本文件 (.txt)
        - 📕 PDF 文件 (.pdf)
        - 📘 Word 文档 (.docx, .doc)
        - 📊 Excel 表格 (.xlsx, .xls)
        - 📋 CSV 文件 (.csv)
        """)
    
    # 显示改写结果（如果有）
    if st.session_state.last_rewritten_query:
        st.markdown("---")
        st.markdown("### 🔄 问题理解")
        st.info(f"📝 **原始:** {st.session_state.messages[-2]['content'] if len(st.session_state.messages) >= 2 else ''}\n\n✍️ **改写:** {st.session_state.last_rewritten_query}")
    
    # 显示检索统计
    if st.session_state.last_retrieved_count > 0:
        st.info(f"🔍 检索到 {st.session_state.last_retrieved_count} 个相关片段")
    
    # 显示知识库统计
    if st.session_state.file_stats:
        st.markdown("---")
        st.markdown("### 📊 知识库统计")
        stats = st.session_state.file_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总文件数", stats.get('total_files', 0))
            st.metric("支持的文件", stats.get('supported_files', 0))
        with col2:
            st.metric("文本块数", stats.get('chunk_count', 0))
            st.metric("不支持的格式", stats.get('unsupported_files', 0))
        
        if stats.get('file_details'):
            with st.expander("📋 已加载文件"):
                for f in stats['file_details']:
                    st.markdown(f"**{f['name']}**")
                    st.markdown(f"大小: {f['cleaned_size']} 字符")
                    st.markdown(f"预览: {f['preview'][:50]}...")
                    st.markdown("---")
    
    st.markdown("---")
    st.markdown("### 💡 使用提示")
    st.markdown("""
    **可以问这些问题试试：**
    - 宋江是怎么死的？
    - 宋江喝毒酒
    - 宋江葬在哪里
    - 第120回讲了什么
    - 武松打虎
    - 林冲风雪山神庙
    """)


# ==================== 主界面 ====================
st.title("📚 水浒传知识问答助手")
st.markdown("---")

# 处理加载按钮点击
if load_button and folder_path:
    if folder_path != st.session_state.current_folder:
        st.session_state.current_folder = folder_path
        with st.spinner("正在初始化知识库，这可能需要几分钟..."):
            chain, retriever_func, stats = load_knowledge_base(folder_path)
            if chain:
                st.session_state.chain = chain
                st.session_state.retriever_func = retriever_func
                st.session_state.file_stats = stats
                st.success(f"✅ 知识库加载成功！")
                st.rerun()
            else:
                st.error("❌ 知识库加载失败，请检查文件夹路径和文件格式")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message and message["references"]:
            with st.expander("📖 查看参考原文"):
                for i, ref in enumerate(message["references"]):
                    st.markdown(f"**来源: {ref['source']}**")
                    st.markdown(f"```\n{ref['content'][:300]}...\n```")
                    if i < len(message["references"]) - 1:
                        st.markdown("---")

# 聊天输入框
if prompt := st.chat_input("请输入您关于水浒传的问题", disabled=not st.session_state.chain):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索相关信息..."):
            try:
                if st.session_state.chain and st.session_state.retriever_func:
                    # 清除之前的改写结果
                    st.session_state.last_rewritten_query = None
                    st.session_state.last_retrieved_count = 0
                    
                    # 检索文档
                    docs = st.session_state.retriever_func(prompt)
                    
                    references = []
                    for doc in docs[:3]:
                        references.append({
                            'source': doc.metadata.get('file_name', '未知来源'),
                            'content': doc.page_content
                        })
                    
                    # 获取回答
                    answer = st.session_state.chain.invoke(prompt)
                    
                    st.markdown(answer)
                    
                    message_data = {"role": "assistant", "content": answer}
                    if references:
                        message_data["references"] = references
                    st.session_state.messages.append(message_data)
                    
                    if references:
                        with st.expander("📖 查看参考原文"):
                            for i, ref in enumerate(references):
                                st.markdown(f"**📄 {ref['source']}**")
                                st.markdown(f"```\n{ref['content'][:300]}...\n```")
                                if i < len(references) - 1:
                                    st.markdown("---")
                    
                    if len(docs) < 2:
                        st.info("💡 只找到少量相关片段，可以尝试换个问法")
                    
                    # 刷新侧边栏显示改写结果
                    st.rerun()
                else:
                    st.error("❌ 知识库未加载，请先在侧边栏加载文件夹")
                    
            except Exception as e:
                st.error(f"❌ 发生错误: {e}")
                import traceback
                with st.expander("查看详细错误"):
                    st.code(traceback.format_exc())

# 欢迎信息
if not st.session_state.chain:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 你好！我是水浒传知识问答助手。
        
        请先在左侧边栏：
        1. 输入包含《水浒传》文本文件的文件夹路径
        2. 点击"加载知识库"按钮
        3. 等待系统处理完文档后，就可以开始提问了
        
        **可以问这些问题试试：**
        - 宋江是怎么死的？
        - 武松打虎的经过
        - 林冲雪夜上梁山
        - 鲁智深倒拔垂杨柳
        - 智取生辰纲的参与者有哪些
        """)


# ==================== 页脚 ====================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 10px;'>"
    "水浒传知识问答系统 | 基于 LangChain + Streamlit + 通义千问"
    "</div>", 
    unsafe_allow_html=True
)
