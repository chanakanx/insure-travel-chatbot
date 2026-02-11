# update_db.py - รวมหลายไฟล์ chunks เพื่อสร้าง/อัปเดต Chroma Vector DB (เหมาะกับ Web Chatbot / RAG)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

# โฟลเดอร์ที่เก็บไฟล์ .jsonl (ค่าเริ่มต้น = โฟลเดอร์เดียวกับไฟล์นี้)
BASE_DIR = Path(__file__).resolve().parent

# รายการไฟล์ chunks ทั้งหมดที่ต้องการรวม (ใส่ชื่อไฟล์ .jsonl ที่อยู่ใน BASE_DIR)
CHUNKS_FILES = [
    "ta_all_in_one_chunks.jsonl",
    "ta_all_in_one_premium.jsonl"  # (ใหม่) ประกัน/FAQ (ตัวอย่าง)
    # ถ้ามีไฟล์อื่นเพิ่มเติม เช่น "promotion_chunks.jsonl" ก็เพิ่มตรงนี้ได้เลย
]

# Debug: แสดงตำแหน่งโฟลเดอร์และไฟล์ .jsonl ที่พบจริง
print(f"📁 BASE_DIR: {BASE_DIR}")
_found_jsonl = sorted([p.name for p in BASE_DIR.glob("*.jsonl")])
print(f"📁 พบไฟล์ .jsonl ในโฟลเดอร์นี้: {_found_jsonl}")

# ถ้าระบุชื่อไฟล์ไว้แต่หาไม่เจอ และในโฟลเดอร์มีไฟล์ .jsonl อื่น ให้ใช้ทั้งหมดแทน
_missing = [fn for fn in CHUNKS_FILES if not (BASE_DIR / fn).exists()]
if _missing and _found_jsonl:
    print(f"⚠️  ไฟล์ที่ระบุไม่พบ: {_missing}")
    print("✅ จะใช้ไฟล์ .jsonl ทั้งหมดที่พบในโฟลเดอร์นี้แทน")
    CHUNKS_FILES = _found_jsonl

persist_directory = str(BASE_DIR / "chroma_db")

# ตั้งค่ารุ่น embedding/collection ให้แก้ได้ง่าย
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "se_life_all")

# ถ้า metadata มี list/dict (เช่น keywords/answer_style) ต้องแปลงเป็น string เพื่อกัน Chroma error
def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            # list/dict/obj อื่น ๆ แปลงเป็น JSON string
            try:
                safe[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                safe[k] = str(v)
    return safe

# ล้าง DB เก่าเพื่อสร้างใหม่ทั้งหมด
if os.path.exists(persist_directory):
    print("🗑️ กำลังล้าง chroma_db เก่า...")
    shutil.rmtree(persist_directory)
os.makedirs(persist_directory, exist_ok=True)  # สร้างโฟลเดอร์ใหม่ (หรือยืนยันว่ามีอยู่)

documents = []
total_loaded = 0

# โหลด embedding
print(f"🤖 กำลังโหลด embedding model ({EMBEDDING_MODEL})...")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ลูปอ่านทุกไฟล์
for file_name in CHUNKS_FILES:
    file_path = (BASE_DIR / file_name).resolve()
    if not file_path.exists():
        print(f"⚠️  ไม่พบไฟล์: {file_name} (มองหา: {file_path}) → ข้ามไป")
        continue

    print(f"📂 กำลังโหลด {file_path.name}...")
    file_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                if "text" not in data:
                    raise KeyError("missing 'text'")

                page_content = data["text"]
                metadata = data.get("metadata", {})

                # เพิ่ม source เพื่อรู้ว่า chunk มาจากไฟล์ไหน
                metadata["source_file"] = file_path.name

                if "id" in data:
                    metadata["id"] = data["id"]

                metadata = sanitize_metadata(metadata)

                documents.append(Document(page_content=page_content, metadata=metadata))
                file_count += 1

            except json.JSONDecodeError as e:
                print(f"   ❌ บรรทัด {line_num} ใน {file_path.name}: JSON error → {e}")
            except Exception as e:
                print(f"   ❌ บรรทัด {line_num} ใน {file_path.name}: {e}")

    print(f"   ✅ โหลดจาก {file_path.name} สำเร็จ: {file_count} chunks")
    total_loaded += file_count

print(f"\n🎉 โหลดข้อมูลรวมทั้งหมด: {total_loaded} chunks")

if total_loaded == 0:
    print("❌ ไม่มีข้อมูลเลย")
    print(f"   - ตรวจสอบว่าไฟล์ .jsonl อยู่ในโฟลเดอร์นี้จริง: {BASE_DIR}")
    print(f"   - ไฟล์ .jsonl ที่พบ: {_found_jsonl}")
    print(f"   - CHUNKS_FILES ที่ใช้งาน: {CHUNKS_FILES}")
    exit(1)

# สร้าง Chroma DB ใหม่
print(f"💾 กำลังสร้างฐานข้อมูลใหม่ใน {persist_directory}...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name=COLLECTION_NAME
)

print(f"✅ อัปเดต Chroma DB สำเร็จ! รวมข้อมูล {total_loaded} ชิ้นเรียบร้อย 🛡️📚✨")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   DB Path: {persist_directory}")
print("   รันคำสั่งนี้เพื่อทดสอบ:")
print("   streamlit run app.py")
print("   หรือ python webhook.py")