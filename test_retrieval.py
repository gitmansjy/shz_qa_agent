import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 设置模型
print("正在加载模型...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("模型加载完成")

# 查找向量数据库目录
vector_dirs = list(Path(".").glob("chroma_db_*"))
if not vector_dirs:
    print("❌ 没有找到向量数据库目录")
    print("请先运行主应用加载知识库")
    exit(1)

print(f"找到向量数据库: {vector_dirs[0]}")

# 连接ChromaDB
try:
    # 尝试不同的ChromaDB版本兼容方式
    client = chromadb.PersistentClient(path=str(vector_dirs[0]))
    
    # 获取所有集合
    collections = client.list_collections()
    if not collections:
        print("❌ 数据库中没有集合")
        exit(1)
    
    collection = collections[0]
    print(f"✅ 成功连接集合: {collection.name}")
    print(f"集合中的文档数量: {collection.count()}")
    
    if collection.count() == 0:
        print("❌ 集合中没有文档，请检查主应用是否成功加载文档")
        exit(1)
        
except Exception as e:
    print(f"❌ 连接数据库失败: {e}")
    exit(1)

# 测试问题列表
test_questions = [
    "宋江",
    "武松",
    "林冲",
    "梁山",
    "打虎",
    "水浒传中谁最厉害"
]

print("\n" + "="*60)
print("开始测试检索功能")
print("="*60)

for question in test_questions:
    print(f"\n📝 问题: {question}")
    print("-"*40)
    
    # 将问题转为向量
    question_embedding = model.encode(question).tolist()
    
    # 检索相似文档
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3  # 返回最相似的3个
    )
    
    # 显示检索结果
    if results['documents'] and results['documents'][0]:
        print(f"✅ 找到 {len(results['documents'][0])} 个相关片段:\n")
        
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            # 相似度分数（distance越小越相似）
            similarity = 1 - distance  # 转换为相似度分数
            
            print(f"--- 片段 {i+1} (相似度: {similarity:.3f}) ---")
            # 显示前200个字符
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(preview)
            print()
    else:
        print("❌ 没有找到相关片段")

print("="*60)
print("测试完成")