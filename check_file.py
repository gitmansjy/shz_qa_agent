# 创建 check_file.py
file_path = "D:/py/study/books/小说/《水浒传》.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    print(f"文件总长度: {len(content)} 字符")
    print(f"前500字符:\n{content[:500]}")
    print(f"后500字符:\n{content[-500:]}")