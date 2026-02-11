import chromadb
from chromadb.utils import embedding_functions

print("Starting check_db.py...")

# ต้องตรงกับ ingest.py เป๊ะ ๆ
CHROMA_PATH = "chroma_db_travel"  # หรือ Path(__file__).parent / "chroma_db_travel"
COLLECTION_NAME = "ta_all_in_one"

# ใช้ embedding เดียวกันกับ ingest.py
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
except Exception:
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

count = collection.count()
print(f"Docs in DB ({COLLECTION_NAME}):", count)

# ถ้าอยากดูรายละเอียดเพิ่ม (optional)
if count > 0:
    sample = collection.peek(limit=1)
    print("Sample metadata:", sample['metadatas'][0] if sample['metadatas'] else "No metadata")
    print("Sample document:", sample['documents'][0][:200] if sample['documents'] else "No document")

print("check_db.py finished.")