import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# โหลด FAQ จาก jsonl
# -----------------------------
docs = []
metadatas = []
ids = []

with open("subacar_allFAQ_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            docs.append(data["text"])
            metadatas.append(data["metadata"])
            ids.append(data["id"])

print(f"โหลด FAQ สำเร็จ: {len(docs)} chunks")

# -----------------------------
# Embedding เดียวกับของรถ
# -----------------------------
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# -----------------------------
# เพิ่มเข้า collection แยก (ในโฟลเดอร์ chroma_db เดียวกัน)
# -----------------------------
faq_vectorstore = Chroma.from_texts(
    texts=docs,
    metadatas=metadatas,
    ids=ids,
    embedding=embedding,
    persist_directory="chroma_db",          # โฟลเดอร์เดียวกับรถยนต์
    collection_name="subacar_faq"           # ชื่อ collection แยก !!!
)

print("เพิ่ม FAQ เข้า Chroma database สำเร็จ (collection: subacar_faq)")

# -----------------------------
# ทดสอบ retrieval
# -----------------------------
retriever = faq_vectorstore.as_retriever(search_kwargs={"k": 5})
test_results = retriever.invoke("ยกเลิกการจองได้ไหม")

print("\n=== ทดสอบ retrieval FAQ ===")
for i, doc in enumerate(test_results):
    print(f"\n{i+1}. {doc.page_content[:300]}...")