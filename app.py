# app.py (เวอร์ชันสมบูรณ์ล่าสุด - รวมทุกการปรับปรุง)
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
import time
import logging
import json
import re
import torch
from pathlib import Path
from dotenv import load_dotenv

# =====================
# Setup
# =====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================
# Abuse Detection
# =====================
def classify_abuse_level(message: str) -> str:
    if not message or not message.strip():
        return "NONE"
    message_lower = message.lower().strip()
    
    if 'บ้าง' in message_lower:
        if re.search(r'(เหี้ย|ควาย|สัส|เย็ด|มึง|กู|แม่ง|ไอ้สัตว์|อีสัตว์|ชิบหาย)', message_lower):
            return "MEDIUM"
        return "NONE"
    
    threat_words = ['ฆ่า', 'ตาย', 'ขู่', 'ทำร้าย', 'ฟ้อง', 'แจ้งความ']
    slander_words = ['โกง', 'หลอก', 'ตุ๋น', 'scam', 'fraud']
    has_threat = any(w in message_lower for w in threat_words)
    has_slander = any(w in message_lower for w in slander_words)
    has_company = any(w in message_lower for w in ['insure', 'อินชัวร์', 'indara', 'บริษัท'])
    
    if has_threat or (has_slander and has_company):
        return "HIGH"
    
    vulgar_words = ['เหี้ย', 'ควาย', 'สัส', 'เย็ด', 'มึง', 'กู', 'แม่ง', 'ชิบหาย', 'ไอ้สัตว์']
    if any(w in message_lower for w in vulgar_words):
        return "MEDIUM"
    
    return "NONE"

# =====================
# Streamlit Config
# =====================
st.set_page_config(page_title="INSURE Travel All in One", page_icon="✈️", layout="centered")
st.title("🛡️ INSURE Travel All in One – ผู้ช่วยประกันเดินทาง")
st.caption("ค่ารักษาพยาบาล • กระเป๋าเดินทาง • ไฟลต์ดีเลย์ • ทรัพย์ในบ้านโจรกรรม | ถามได้เลยค่ะ ✈️")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0.0

# =====================
# Helper Functions
# =====================
def safe_get_question(x):
    q = x.get("question", "")
    return str(q) if q else ""

def format_docs(docs):
    if not docs:
        return ""
    formatted = "\n\n".join(doc.page_content for doc in docs if hasattr(doc, "page_content"))
    return formatted

def extract_trip_slots(text: str) -> dict:
    text = text.lower()
    slots = {}
    if re.search(r"\d+\s*วัน", text):
        slots["days"] = re.search(r"(\d+)\s*วัน", text).group(1)
    if any(w in text for w in ["ญี่ปุ่น", "เกาหลี", "ยุโรป", "เชงเก้น", "อเมริกา", "ออสเตรเลีย"]):
        slots["destination"] = "ต่างประเทศ"
    if re.search(r"\d+\s*คน", text):
        slots["people"] = re.search(r"(\d+)\s*คน", text).group(1)
    return slots

def simple_recommend(trip_slots: dict):
    if not trip_slots or "destination" not in trip_slots:
        return {
            "recommended_plans": [],
            "missing_slots": ["destination"],
            "next_question": "ไปเที่ยวประเทศไหนคะ?"
        }

    plans = [
        {"plan_id": "plan3", "label": "แผน 3", "reason": "คุ้มครองครบ เหมาะทริปทั่วไป"},
        {"plan_id": "plan5", "label": "แผน 5", "reason": "วงเงินสูงสุด เหมาะทริปต่างประเทศ"}
    ]
    return {
        "recommended_plans": plans,
        "missing_slots": [],
        "next_question": ""
    }

# =====================
# Load Components
# =====================
@st.cache_resource
def load_components():
    embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}   # แนะนำเปิดสำหรับ E5 series
)
    persist_dir = str(Path(__file__).resolve().parent / "chroma_db_travel")

    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="ta_all_in_one"
        )
        logger.info(f"โหลด Chroma สำเร็จ: {persist_dir}")
    except Exception as e:
        logger.error(f"โหลด Chroma ไม่ได้: {e}")
        st.error("ไม่สามารถโหลดฐานข้อมูลได้ กรุณารัน reset_and_ingest.py ก่อน")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # BUSINESS_PROMPT (เดิม + ปรับให้ยืดหยุ่น)
    BUSINESS_PROMPT = PromptTemplate.from_template("""\
คุณคือผู้ช่วยแนะนำประกันเดินทางของบริษัท (Travel Insurance Assistant)
ตอบภาษาไทยเท่านั้น สุภาพ กระชับ อ่านง่าย

กติกาสำคัญ:
- ใช้เฉพาะข้อมูลจาก context เท่านั้น ห้ามแต่ง/เดา/อนุมานรายละเอียดความคุ้มครองหรือเงื่อนไข
- ห้ามรับประกันผลการเคลม หรือบอกว่า “เคลมได้แน่นอน”
- ถ้า context ไม่พบข้อมูลจริง ๆ ให้ตอบว่า: "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"
- ความยาวคำตอบรวมไม่เกิน 650 ตัวอักษร
- ถ้าผู้ใช้ถามเชิงเปรียบเทียบ ให้เปรียบเทียบได้ไม่เกิน 3 แผน และต้องอิงจาก context เท่านั้น
- หากข้อมูลที่ผู้ใช้ให้มายังไม่พอสำหรับแนะนำแผน ให้ถามกลับ 1 ข้อ (ห้ามถามเป็นชุดแบบฟอร์ม)

ข้อมูลจากฐานข้อมูลจริง (กรมธรรม์/FAQ/สรุปแผน):
{context}

ประวัติการสนทนาก่อนหน้า:
{chat_history}
(ใช้เพื่อไม่ถามซ้ำ และคงบริบทเดิม ห้ามสร้างข้อมูลใหม่จาก chat_history)

ข้อความผู้ใช้ล่าสุด:
{question}

แนวทางตอบ:
1) ถ้าคำถามเป็น FAQ/เงื่อนไข → ตอบตรงคำถามแบบสั้น ๆ
2) ถ้าคำถามเป็น “อยากได้แผนที่เหมาะ” → สรุป 1–3 แผนที่เหมาะ (ถ้ามีข้อมูลใน context)
3) ปิดท้ายด้วยคำถามต่อยอด 1 ข้อ เฉพาะข้อมูลที่ยังขาด (ถ้าจำเป็น)
4) หากมีลิงก์ซื้อที่ระบบให้มาในตัวแปร {buy_link} ให้แสดงท้ายสุด 1 บรรทัด:
"ไปซื้อออนไลน์ได้ที่: {buy_link}"

ตอบเลย:
""")

    # RECOMMEND_PROMPT (เดิม + ปรับให้ถามต่อเนื่อง)
    RECOMMEND_PROMPT = PromptTemplate.from_template("""\
คุณคือผู้ช่วยแนะนำแผนประกันเดินทาง Travel All in One  
ตอบภาษาไทย กระชับ ชัดเจน ช่วยตัดสินใจ และพาลูกค้าไปขั้นถัดไปอย่างเป็นธรรมชาติ

กติกา:
- ใช้เฉพาะข้อมูลจาก {context} และ {recommended_plans_json} เท่านั้น ห้ามแต่งจุดเด่นหรือเงื่อนไข
- ห้ามรับประกันผลเคลมหรือบอกว่าแผนไหน “ดีที่สุด” แบบเด็ดขาด
- ใช้ {chat_history} เพื่อไม่ถามซ้ำ และต่อยอดจากสิ่งที่ลูกค้าเพิ่งบอก

การตอบ:
- ถ้าข้อมูลทริปยังไม่ครบ → ถามกลับ**เพียง 1 ข้อ** ที่จำเป็นที่สุด (เลือกจากลำดับด้านล่าง)
- ถ้าข้อมูลพอ → แนะนำ 1–2 แผนจาก recommended_plans_json  
  อธิบายเหตุผลสั้น ๆ ให้สอดคล้องกับ trip_slots + ข้อมูลใน context  
  ถ้ามีเบี้ยใน context ให้ระบุเบี้ยโดยประมาณตามเงื่อนไข (โซน / วัน / ประเภท)

ลำดับคำถาม (ถามทีละ 1 ข้อ):
1. ปลายทางเดินทางแน่นอนหรือยังคะ
2. ทริปนี้เดินทางกี่วัน
3. เดินทางทั้งหมดกี่ท่าน
4. มีกิจกรรมพิเศษหรือเสี่ยงไหมคะ (เช่น ดำน้ำ สกี มอเตอร์ไซค์)
5. อยากให้เน้นคุ้มครองเรื่องใดเป็นพิเศษไหม

รูปแบบการตอบแนะนำ (เมื่อข้อมูลพอ):
• แผน X – เหมาะกับ … เพราะ …
  จุดเด่นหลัก: … / …
  เบี้ยโดยประมาณ: … บาท (เงื่อนไข …)
• แผน Y – เหมาะถ้าต้องการ …

จากนั้นปิดด้วยประโยคนำ + คำถาม 1 ข้อ (ถ้ายังมีจุดที่ต้องชี้แจง)

ข้อมูลทริป:
{trip_slots}

แผนแนะนำจากระบบ:
{recommended_plans_json}

ข้อมูลจากฐาน:
{context}

ประวัติการคุย:
{chat_history}

คำถามปัจจุบัน:
{question}

ตอบเลย:
""")

    # Chains
    business_chain = (
        {
            "context": RunnableLambda(safe_get_question) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(safe_get_question),
            "chat_history": lambda x: x.get("chat_history", ""),
            "buy_link": lambda x: x.get("buy_link", "https://www.indara.co.th/products/travel-insurance/ta-allinone")
        }
        | BUSINESS_PROMPT
        | llm
        | StrOutputParser()
    )

    recommend_chain = (
        {
            "trip_slots": lambda x: x["trip_slots"],
            "recommended_plans_json": lambda x: x["recommended_plans_json"],
            "context": RunnableLambda(safe_get_question) | retriever | RunnableLambda(format_docs),
            "chat_history": lambda x: x.get("chat_history", ""),
            "question": RunnableLambda(safe_get_question)
        }
        | RECOMMEND_PROMPT
        | llm_creative
        | StrOutputParser()
    )

    return business_chain, recommend_chain, retriever

# โหลด
with st.spinner("กำลังโหลดฐานข้อมูล..."):
    business_chain, recommend_chain, retriever = load_components()
st.success("พร้อมใช้งานแล้วค่ะ ✈️")

# Chat UI
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("ถามเรื่องประกันเดินทางได้เลยค่ะ ✈️"):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("กำลังตอบ..."):
            start = time.time()

            try:
                history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-8:])

                # Abuse Detection
                if classify_abuse_level(user_input) != "NONE":
                    response = "ขออภัยนะคะ ระบบไม่สามารถตอบข้อความที่มีถ้อยคำไม่เหมาะสมได้ค่ะ\nกรุณาถามใหม่ด้วยคำสุภาพนะคะ 😊"
                    mode = "ESCALATION"
                else:
                    recommend_keywords = ["แนะนำ", "ควรซื้อ", "แผนไหน", "เหมาะกับ", "ไปญี่ปุ่น", "ไปเกาหลี", "ต่างประเทศ"]
                    is_recommend = any(kw in user_input.lower() for kw in recommend_keywords)

                    trip_slots = extract_trip_slots(user_input)
                    recommend_info = simple_recommend(trip_slots)

                    with get_openai_callback() as cb:
                        if is_recommend or recommend_info["recommended_plans"]:
                            mode = "RECOMMEND"
                            response = recommend_chain.invoke({
                                "trip_slots": json.dumps(trip_slots, ensure_ascii=False),
                                "recommended_plans_json": json.dumps(recommend_info["recommended_plans"], ensure_ascii=False),
                                "chat_history": history,
                                "question": user_input
                            })
                        else:
                            mode = "BUSINESS"
                            response = business_chain.invoke({
                                "question": user_input,
                                "chat_history": history,
                                "buy_link": "https://www.indara.co.th/products/travel-insurance/ta-allinone"
                            })

                    st.session_state.total_tokens += cb.total_tokens
                    st.session_state.total_cost += cb.total_cost

                st.markdown(response)
                st.caption(f"ใช้เวลา {time.time() - start:.2f} วินาที | Mode: {mode}")
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                logger.exception("Error")
                st.error(f"เกิดข้อผิดพลาด: {str(e)}\nกรุณาลองใหม่อีกครั้งนะคะ")

# Sidebar
with st.sidebar:
    st.header("ข้อมูลการใช้งาน")
    st.write(f"Tokens: {st.session_state.total_tokens:,}")
    st.write(f"Cost ≈ ${st.session_state.total_cost:.6f}")
    if st.button("ล้างประวัติ"):
        st.session_state.chat_history = []
        st.rerun()
