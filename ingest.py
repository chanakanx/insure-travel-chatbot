# ingest.py (เวอร์ชันแก้ไข)

import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

print("Starting ingestion...")

jsonl_path = Path("ta_all_in_one_chunks.jsonl")
persist_dir = Path("chroma_db_travel")

if not jsonl_path.exists():
    print(f"Error: ไม่พบไฟล์ {jsonl_path}")
    exit(1)

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            text = data["text"]
            metadata = data.get("metadata", {})

            # แปลงทุก field ที่เป็น list ให้เป็น string เพื่อให้ Chroma รับได้
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = str(value)  # เช่น "[1, 2]" หรือ "[2, 3]"
                elif isinstance(value, dict):
                    metadata[key] = json.dumps(value)  # ถ้ามี dict ก็แปลงเป็น string

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            print(f"Error parsing line: {e}")

print(f"Loaded {len(documents)} documents from JSONL")

vectorstore = Chroma(
    persist_directory=str(persist_dir),
    embedding_function=embedding,
    collection_name="ta_all_in_one"
)

# ลบข้อมูลเก่าทั้งหมดก่อน (optional ถ้าอยากเริ่มใหม่)
# vectorstore.delete_collection()

if documents:
    vectorstore.add_documents(documents)
    print(f"Added/Updated {len(documents)} documents to collection 'ta_all_in_one'")
else:
    print("No documents to add")

count = vectorstore._collection.count()
print(f"Total docs in DB after ingestion: {count}")

print("Ingestion finished.")