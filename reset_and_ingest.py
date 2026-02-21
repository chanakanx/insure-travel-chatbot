# reset_and_ingest.py (เวอร์ชันแก้ metadata list error 100% - กรองเอง)
import os
import shutil
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_metadata(metadata):
    """กรอง metadata ให้เหลือเฉพาะ str, int, float, bool, None"""
    if not isinstance(metadata, dict):
        return {}
    
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            cleaned[key] = value
        elif isinstance(value, (list, dict)):
            # แปลง list/dict เป็น string เพื่อเก็บไว้ได้
            cleaned[key] = json.dumps(value, ensure_ascii=False)
        else:
            logger.warning(f"ข้าม metadata key '{key}' เพราะค่าเป็น {type(value)}")
    return cleaned

def reset_chroma_and_ingest():
    base_dir = Path(__file__).resolve().parent
    persist_dir = base_dir / "chroma_db_travel"
    jsonl_path = base_dir / "ta_all_in_one_chunks.jsonl"

    # 1. ลบ Chroma DB เก่า
    if persist_dir.exists():
        logger.info(f"กำลังลบฐานข้อมูลเก่าที่: {persist_dir}")
        shutil.rmtree(persist_dir)
        logger.info("ลบฐานข้อมูลเก่าเรียบร้อย")
    else:
        logger.info("ไม่พบฐานข้อมูลเก่า")

    # 2. ตรวจสอบไฟล์ JSONL
    if not jsonl_path.exists():
        logger.error(f"ไม่พบไฟล์: {jsonl_path}")
        return

    # 3. อ่าน JSONL + สร้าง Document + กรอง metadata
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    logger.warning(f"บรรทัด {line_num}: ไม่ใช่ JSON object → ข้าม")
                    continue

                text = data.get('text', '')
                if not text:
                    logger.warning(f"บรรทัด {line_num}: ไม่มี 'text' → ข้าม")
                    continue

                metadata = data.get('metadata', {})
                cleaned_metadata = clean_metadata(metadata)

                doc = Document(page_content=text, metadata=cleaned_metadata)
                documents.append(doc)

            except json.JSONDecodeError as e:
                logger.warning(f"บรรทัด {line_num}: parse JSON ไม่ได้ → {e}")
            except Exception as e:
                logger.warning(f"บรรทัด {line_num}: มีปัญหา → {e}")

    logger.info(f"อ่านข้อมูลสำเร็จ {len(documents)} chunks")

    if not documents:
        logger.error("ไม่มีข้อมูล Document ที่ถูกต้อง → หยุดการทำงาน")
        return

    # 4. Ingest เข้า Chroma ใหม่
    logger.info("กำลังสร้างฐานข้อมูลใหม่และ ingest...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name="ta_all_in_one"
        )

        count = vectorstore._collection.count()
        logger.info(f"Ingest สำเร็จ! จำนวน documents ในฐานข้อมูล: {count}")

        print("\n" + "═" * 70)
        print("รีเซ็ต Chroma + Ingest ข้อมูลใหม่เรียบร้อยแล้ว!")
        print(f"ฐานข้อมูลใหม่มี {count:,} chunks")
        print(f"ตำแหน่ง: {persist_dir}")
        print("พร้อมรัน app.py แล้วครับ ✈️")
        print("═" * 70 + "\n")

    except Exception as e:
        logger.error(f"Ingest ล้มเหลว: {e}")
        print("แนะนำ: เปิดไฟล์ ta_all_in_one_chunks.jsonl แล้วตรวจสอบว่า")
        print("ทุกบรรทัดเป็น JSON object เดียว ไม่มี comma ท้ายบรรทัด หรือบรรทัดว่างผิดปกติ")

if __name__ == "__main__":
    print("=== รีเซ็ต Chroma แล้ว ingest ข้อมูลใหม่ (แก้ metadata list error) ===\n")
    reset_chroma_and_ingest()