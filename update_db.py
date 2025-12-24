# update_db.py - รองรับหลายไฟล์ chunks (รถ + FAQ + Service) เวอร์ชันสมบูรณ์ 100%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import os
import shutil

# รายการไฟล์ chunks ทั้งหมดที่ต้องการรวม
CHUNKS_FILES = [
    "subacar_chunks.jsonl",           # รถ 24 คัน (ไฟล์เดิม)
    "subacar_allFAQ_chunks.jsonl"     # FAQ + Service (ไฟล์ใหม่ที่คุณมี)
    # ถ้ามีไฟล์อื่นเพิ่มเติม เช่น "promotion_chunks.jsonl" ก็เพิ่มตรงนี้ได้เลย
]

persist_directory = "chroma_db"

# ล้าง DB เก่าเพื่อสร้างใหม่ทั้งหมด
if os.path.exists(persist_directory):
    print("🗑️ กำลังล้าง chroma_db เก่า...")
    shutil.rmtree(persist_directory)
    os.makedirs(persist_directory)  # สร้างโฟลเดอร์ว่างใหม่

documents = []
total_loaded = 0

# โหลด embedding
print("🤖 กำลังโหลด embedding model (BAAI/bge-m3)...")
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# ลูปอ่านทุกไฟล์
for file_name in CHUNKS_FILES:
    if not os.path.exists(file_name):
        print(f"⚠️  ไม่พบไฟล์: {file_name} → ข้ามไป")
        continue
    
    print(f"📂 กำลังโหลด {file_name}...")
    file_count = 0
    
    with open(file_name, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                page_content = data["text"]
                metadata = data.get("metadata", {})
                
                # เพิ่ม source เพื่อรู้ว่า chunk มาจากไฟล์ไหน
                metadata["source_file"] = file_name
                
                if "id" in data:
                    metadata["id"] = data["id"]
                
                documents.append(Document(page_content=page_content, metadata=metadata))
                file_count += 1
                
            except json.JSONDecodeError as e:
                print(f"   ❌ บรรทัด {line_num} ใน {file_name}: JSON error → {e}")
            except Exception as e:
                print(f"   ❌ บรรทัด {line_num} ใน {file_name}: {e}")
    
    print(f"   ✅ โหลดจาก {file_name} สำเร็จ: {file_count} chunks")
    total_loaded += file_count

print(f"\n🎉 โหลดข้อมูลรวมทั้งหมด: {total_loaded} chunks")

if total_loaded == 0:
    print("❌ ไม่มีข้อมูลเลย กรุณาตรวจสอบชื่อไฟล์และวางไฟล์ในโฟลเดอร์นี้")
    exit(1)

# สร้าง Chroma DB ใหม่
print(f"💾 กำลังสร้างฐานข้อมูลใหม่ใน {persist_directory}...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name="subacar_all"  # ชื่อ collection ใหม่ (รวมทุกอย่าง)
)

print(f"✅ อัปเดต Chroma DB สำเร็จ! รวมข้อมูล {total_loaded} ชิ้นเรียบร้อย 🚗📋✨")
print("   รันคำสั่งนี้เพื่อทดสอบ:")
print("   streamlit run app.py")
print("   หรือ python webhook.py")