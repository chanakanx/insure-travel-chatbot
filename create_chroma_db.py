import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# โหลดข้อมูลรถยนต์จาก subacar_chunks.jsonl
# -----------------------------
docs = []        # ข้อความที่จะ embed
metadatas = []   # metadata
ids = []         # id unique

print("กำลังโหลดข้อมูลรถยนต์...")

with open("subacar_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            docs.append(data["text"])
            metadatas.append(data["metadata"])
            ids.append(data["id"])

print(f"โหลดข้อมูลรถยนต์สำเร็จ: {len(docs)} chunks")

# -----------------------------
# โหลดข้อมูล FAQ และบริการจาก subacar_allFAQ_chunks.jsonl
# -----------------------------
print("กำลังโหลดข้อมูล FAQ และบริการ...")

with open("subacar_allFAQ_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            docs.append(data["text"])
            metadatas.append(data["metadata"])
            ids.append(data["id"])

print(f"โหลดข้อมูล FAQ สำเร็จ: {len(docs) - (len(docs) - len([d for d in docs[-len(metadatas):] if 'faq' in d.lower() or 'service' in d.lower()]))} chunks เพิ่มเติม")
print(f"รวมทั้งหมด: {len(docs)} chunks")

# -----------------------------
# สร้าง Embedding model (เดียวกับใน app.py)
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# -----------------------------
# สร้างหรืออัปเดต Chroma database (collection เดียว)
# -----------------------------
print("กำลังสร้าง Chroma database... (อาจใช้เวลา 1-3 นาที ขึ้นกับเครื่อง)")

vectorstore = Chroma.from_texts(
    texts=docs,
    metadatas=metadatas,
    ids=ids,
    embedding=embedding,
    persist_directory="chroma_db",
    collection_name="subacar_all"  # ชื่อ collection เดียว รองรับทั้งรถและ FAQ
)

print("สร้างและบันทึก Chroma database สำเร็จ!")
print("โฟลเดอร์: chroma_db")
print("Collection: subacar_all")

# -----------------------------
# ทดสอบ retrieval (ตัวอย่าง)
# -----------------------------
print("\n=== ทดสอบการค้นหา ===")

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ทดสอบเรื่องรถไฟฟ้า
results = retriever.invoke("รถไฟฟ้ามีให้เช่าไหม")
print("\n1. ทดสอบ query: 'รถไฟฟ้ามีให้เช่าไหม'")
for i, doc in enumerate(results[:3]):
    title = doc.metadata.get("title", doc.metadata.get("question", "ไม่ระบุ"))
    print(f"   - {i+1}. {title}")

# ทดสอบเรื่องยกเลิกการจอง
results = retriever.invoke("ยกเลิกการจองได้ไหม")
print("\n2. ทดสอบ query: 'ยกเลิกการจองได้ไหม'")
for i, doc in enumerate(results[:3]):
    title = doc.metadata.get("title", doc.metadata.get("question", "ไม่ระบุ"))
    print(f"   - {i+1}. {title}")

# ทดสอบเรื่องติดต่อ
results = retriever.invoke("ติดต่อ subacar ยังไง")
print("\n3. ทดสอบ query: 'ติดต่อ subacar ยังไง'")
for i, doc in enumerate(results[:3]):
    title = doc.metadata.get("title", doc.metadata.get("question", "ไม่ระบุ"))
    print(f"   - {i+1}. {title}")

print("\nเสร็จสิ้น! ลองรัน streamlit run app.py ได้เลยครับ 🚗💬")