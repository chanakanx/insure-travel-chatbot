from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json

print("กำลังโหลดข้อมูลรถ 24 คัน...")
chunks = []
with open("subacar_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

texts = [c["text"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]

print(f"พบรถ {len(chunks)} คัน กำลังสร้างฐานข้อมูล...")

# ใช้ bge-m3 → ดีที่สุดสำหรับภาษาไทย + multilingual
vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),  # ตัวนี้ผ่านแน่นอน
    collection_name="subacar",
    persist_directory="chroma_db"
)

print("สร้าง Chroma DB เสร็จเรียบร้อยแล้ว!")
print("รันคำสั่งนี้ได้เลย → streamlit run app.py")