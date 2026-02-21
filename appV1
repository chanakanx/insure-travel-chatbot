# app.py (เวอร์ชันแก้ไขล่าสุด - เพิ่ม abuse detection & escalation)
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
    """
    ตรวจจับระดับความไม่เหมาะสมของข้อความผู้ใช้
    เวอร์ชันปรับปรุง: ดักคำว่า "บ้าง" ก่อน เพื่อป้องกัน false positive
    return: "LOW", "MEDIUM", "HIGH", หรือ "NONE"
    """
    if not message or not message.strip():
        return "NONE"
    
    message_lower = message.lower().strip()
    
    # === ดักกรณีคำถามที่มี "บ้าง" ก่อนเลย (ส่วนใหญ่เป็นคำถามสุภาพ) ===
    if 'บ้าง' in message_lower:
        # ยกเว้นเฉพาะกรณีที่มีคำหยาบรุนแรงนำหน้าชัดเจน
        if re.search(r'(เหี้ย|ควาย|สัส|เย็ด|มึง|กู|แม่ง|ไอ้สัตว์|อีสัตว์|ชิบหาย)', message_lower):
            return "MEDIUM"
        else:
            return "NONE"
    
    # === HIGH: คุกคามหรือใส่ร้ายบริษัท ===
    threat_words = ['ฆ่า', 'ตาย', 'ขู่', 'ทำร้าย', 'ฟ้อง', 'แจ้งความ', 'ทำลาย', 'เผา', 'จัดการ', 'กรูจะ']
    slander_words = ['โกง', 'หลอก', 'ตุ๋น', 'ฉ้อโกง', 'scam', 'fraud', 'หลอกลวง', 'โจร']
    has_threat = any(word in message_lower for word in threat_words)
    has_slander = any(word in message_lower for word in slander_words)
    has_company = any(k in message_lower for k in ['in sure', 'insure', 'อินชัว', 'บริษัท', 'พวกมึง', 'พวกนี้', 'indara', 'อินทาระ'])
    
    if has_threat or (has_slander and has_company):
        return "HIGH"
    
    # === MEDIUM: คำหยาบหรือด่าชัดเจน ===
    vulgar_words = [
        'เหี้ย', 'ไอ้เหี้ย', 'ควาย', 'สัส', 'เย็ด', 'มึง', 'กู', 'แม่ง', 'ชิบหาย',
        'ไอ้สัตว์', 'อีสัตว์', 'อีดอก', 'ส้นตีน', 'หน้าตัวเมีย', 'เหี้ยเอ้ย',
        'ไอ้', 'ไอ', 'เวร', 'สัตว์', 'เพี้ยน', 'หน้าตัว'
    ]
    strong_bully_words = [
        'โง่', 'ปัญญาอ่อน', 'งี่เง่า', 'หน้าเงิน', 'เลว', 'ห่วยแตก', 'ขี้ข้า'
    ]
    
    has_vulgar = any(word in message_lower for word in vulgar_words)
    has_strong_bully = any(word in message_lower for word in strong_bully_words)
    
    if has_vulgar or has_strong_bully:
        return "MEDIUM"
    
    # === LOW: น้ำเสียงไม่สุภาพเล็กน้อย ===
    mild_negative_words = ['ห่วย', 'แย่', 'ขี้เกียจ', 'ไร้สาระ', 'ด้อย', 'เสียดาย']
    excessive_exclamation = re.search(r'[!?]{6,}', message)
    rude_ending = re.search(r'\bวะ\b|\bเว้ย\b|\bเฮ้ย\b', message_lower)
    
    if any(word in message_lower for word in mild_negative_words) or excessive_exclamation or rude_ending:
        return "LOW"
    
    return "NONE"

# =====================
# Escalation Chain
# =====================
fallback_escalation_template = """
คุณคือระบบป้องกันความสุภาพของ Chatbot
หน้าที่หลักคือตอบกลับเมื่อผู้ใช้ใช้ถ้อยคำไม่เหมาะสมเท่านั้น

ใช้ Template นี้เมื่อ classify_abuse_level ตรวจพบระดับ LOW / MEDIUM / HIGH เท่านั้น
ห้ามใช้กับข้อความที่สุภาพหรือเป็นคำถามปกติ

────────────────────────
ข้อมูลสำคัญที่ต้องรู้:
ระดับความรุนแรงที่ตรวจพบจากระบบ: {level}
ข้อความผู้ใช้จริง: {question}

────────────────────────
กติกาเข้มงวด (ห้ามละเมิด):
- ต้องตอบตามระดับที่ระบุใน {level} เท่านั้น ห้ามเปลี่ยนระดับเอง
- ห้ามตอบโต้ ห้ามประชด ห้ามตำหนิ ห้ามซ้ำคำหยาบของผู้ใช้
- ห้ามแสดงความเห็นส่วนตัว
- ใช้ภาษาสุภาพ เป็นกลาง ตลอดเวลา
- ความยาวสั้น กระชับ
- ห้ามตอบข้อมูลบริการหรือเชื่อมต่อ RAG/FAQ ในระดับ MEDIUM และ HIGH
- ระดับ LOW เท่านั้นที่อาจตอบข้อมูลบริการได้ (ถ้ามีคำถามชัดเจน)

────────────────────────
คำตอบที่ต้องใช้ตามระดับ (เลือกตาม {level} เท่านั้น):

ถ้า {level} = LOW
→ ตอบคำถามหลักตามปกติก่อน (ถ้ามี) แล้วเพิ่มเตือนสุภาพ 1 ประโยค
ตัวอย่าง:
[ตอบข้อมูลตามคำถามปกติก่อน]
ขอความร่วมมือใช้ถ้อยคำที่สุภาพในการสนทนานะคะ 😊
หากมีคำถามเกี่ยวกับประกันเดินทาง สามารถสอบถามใหม่ได้เลยค่ะ

ถ้า {level} = MEDIUM
→ ตั้งขอบเขตชัดเจน ห้ามตอบเนื้อหาคำถาม
ต้องตอบข้อความนี้เท่านั้น:
ขออภัยนะคะ ระบบไม่สามารถตอบข้อความที่มีถ้อยคำไม่เหมาะสมได้ค่ะ
ขอความร่วมมือใช้ภาษาที่สุภาพในการสนทนานะคะ
หากมีคำถามเกี่ยวกับประกันเดินทาง สามารถสอบถามใหม่ได้เลยค่ะ

ถ้า {level} = HIGH
→ ยุติการสนทนาทันที ห้ามชวนคุยต่อ
ต้องตอบข้อความนี้เท่านั้น:
ขออภัยค่ะ ระบบไม่สามารถดำเนินการสนทนาที่มีถ้อยคำไม่เหมาะสมต่อได้ค่ะ
หากต้องการความช่วยเหลือ สามารถติดต่อเจ้าหน้าที่ผ่านช่องทางที่ระบุบนเว็บไซต์ได้เลยนะคะ

────────────────────────
ข้อห้ามเด็ดขาด:
- ห้ามตอบว่า "เข้าใจแล้วค่ะ" หรือ "ยินดีช่วยเสมอ" ในระดับ MEDIUM/HIGH
- ห้ามเปิดช่องให้ผู้ใช้ต่อความไม่สุภาพ
- ห้ามให้ข้อมูลบริการในระดับ MEDIUM และ HIGH

ตอบเลยตามระดับที่ระบุใน {level} เท่านั้น:
"""

# =====================
# Streamlit Config & Session
# =====================
st.set_page_config(
    page_title="INSURE Travel All in One",
    page_icon="✈️",
    layout="centered"
)

st.title("INSURE Travel All in One – ประกันเดินทาง")
st.caption("ค่ารักษาพยาบาล • กระเป๋าเดินทาง • ไฟลต์ดีเลย์ • ทรัพย์ในบ้านโจรกรรม | ถามได้เลยค่ะ ✈️")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0.0

# =====================
# Helper Functions (เดิม)
# =====================
def safe_get_question(x):
    q = x.get("question", "")
    if isinstance(q, dict):
        logger.warning("question เป็น dict → แปลงเป็น string: %s", q)
        return str(q)
    if isinstance(q, (list, tuple)):
        return " ".join(str(item) for item in q)
    return str(q) if q else ""

def format_docs(docs):
    if not docs:
        logger.info("No relevant context retrieved")
        return "ไม่มีข้อมูลที่เกี่ยวข้องในฐานข้อมูลค่ะ"
    if isinstance(docs, dict):
        logger.warning("docs เป็น dict แทน list: %s", docs)
        return str(docs)
    
    formatted = "\n\n".join(
        doc.page_content for doc in docs
        if hasattr(doc, "page_content") and isinstance(doc.page_content, str)
    )
    logger.info(f"Retrieved context length: {len(formatted)} chars")
    return formatted

def extract_trip_slots(text: str) -> dict:
    text = text.lower()
    slots = {}
    if re.search(r"\d+\s*วัน", text):
        match = re.search(r"(\d+)\s*วัน", text)
        if match:
            slots["days"] = match.group(1)
    if any(w in text for w in ["ยุโรป", "เชงเก้น", "ญี่ปุ่น", "เกาหลี", "จีน", "อเมริกา", "ออสเตรเลีย", "สิงคโปร์"]):
        slots["destination"] = "ต่างประเทศ"
    if re.search(r"\d+\s*คน", text):
        match = re.search(r"(\d+)\s*คน", text)
        if match:
            slots["people"] = match.group(1)
    return slots

# =====================
# Load Components
# =====================
@st.cache_resource
def load_components():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    base_dir = Path(__file__).resolve().parent
    persist_dir = str(base_dir / "chroma_db_travel")

    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="ta_all_in_one"
        )
        logger.info(f"โหลด Chroma DB สำเร็จ: {persist_dir}")
    except Exception as e:
        logger.error(f"โหลด Chroma ไม่ได้: {e}")
        st.error("ไม่สามารถโหลดฐานข้อมูลได้ กรุณาตรวจสอบโฟลเดอร์ chroma_db_travel")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # BUSINESS_PROMPT (เดิม + ปรับให้ยืดหยุ่น)
    BUSINESS_PROMPT = PromptTemplate.from_template("""
คุณคือผู้ช่วยประกันเดินทาง Travel All in One ของ INSURE
ตอบภาษาไทย สุภาพ กระชับ อ่านง่าย

กติกาสำคัญ:
- ใช้เฉพาะข้อมูลจาก context เท่านั้น ห้ามแต่ง ห้ามเดา ห้ามอนุมาน
- ห้ามรับประกันผลการเคลม หรือบอกว่าเคลมได้แน่นอน
- ถ้าคำถามเกี่ยวกับ "จุดเด่น" "จุดเด่นของประกัน" "ข้อดี" "ดีตรงไหน" "สรุปจุดเด่น"  
  ให้สรุปจากส่วนที่ขึ้นต้นด้วย "จุดเด่นของประกัน" ใน context ก่อนเป็นอันดับแรก ถ้าเจอให้ตอบตามนั้นเลย
- ถ้าคำถามเป็น "แนะนำแผน" "ควรซื้อแผนไหน" "ไป [ประเทศ] ควรซื้อแผนอะไร" "ประกันไปญี่ปุ่น" หรือคำถามเชิงแนะนำ  
  ให้สรุปความคุ้มครองหลักจาก context (เช่น แผนสูงสุดเหมาะต่างประเทศ) และถามกลับข้อมูลที่จำเป็น 1 ข้อ (เช่น กี่วัน, มีกิจกรรมเสี่ยงไหม)
- ถ้าคำถามเกี่ยวกับ "ราคา" "เบี้ย" "เบี้ยประกัน" "ค่าใช้จ่าย" ให้ตอบตามตารางเบี้ยใน context โดยตรง (ระบุ Asia/Worldwide, ระยะเวลา, รวมอากรแสตมป์แล้ว)
- ถ้า context ว่างหรือไม่เกี่ยวข้องจริง ๆ เท่านั้น ให้ตอบ "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"
- ความยาวไม่เกิน 650 ตัวอักษร

ข้อมูลจากฐานข้อมูลจริง:
{context}

ประวัติการสนทนาก่อนหน้า:
{chat_history}

คำถามผู้ใช้ล่าสุด:
{question}

ถ้ามีลิงก์ซื้อออนไลน์ ให้แสดงท้ายสุด:
"ซื้อออนไลน์ได้ที่: {buy_link}"

ตอบเลย:
""")

    # RECOMMEND_PROMPT (เดิม + ปรับให้ถามต่อเนื่อง)
    RECOMMEND_PROMPT = PromptTemplate.from_template("""
คุณคือผู้ช่วยแนะนำแผนประกันเดินทาง
ตอบภาษาไทย กระชับ ช่วยตัดสินใจ

กติกา:
- ใช้ข้อมูลจาก context และ recommended_plans_json เท่านั้น ห้ามแต่งจุดเด่น
- ห้ามรับประกันเคลมหรือการันตี
- ถ้าข้อมูลทริปไม่ครบ ให้ถามกลับเฉพาะข้อสำคัญ 1 ข้อ (เช่น กี่วัน, ปลายทางแน่นอนไหม)
- ให้เหตุผลตาม trip_slots และ context
- ถ้ามีข้อมูลเบี้ยใน context ให้ระบุเบี้ยโดยประมาณตามแผนและระยะเวลา (รวมอากรแสตมป์)

ข้อมูลทริป:
{trip_slots}

แผนแนะนำ:
{recommended_plans_json}

ข้อมูลจากฐาน:
{context}

ประวัติการคุย:
{chat_history}

คำถาม:
{question}

รูปแบบ:
- bullet
- แต่ละแผน: ชื่อ • เหมาะกับ • จุดเด่นหลัก 1-2 ข้อ • เบี้ยประมาณ (ถ้ามี)
- ปิดด้วยคำถามสุภาพ เช่น "สนใจแผนไหนเป็นพิเศษไหมคะ" หรือ "เดินทางกี่วันคะ?"

ตอบเลย:
""")

    # Escalation Chain
    escalation_prompt = PromptTemplate.from_template(fallback_escalation_template)
    escalation_chain = escalation_prompt | llm | StrOutputParser()

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

    return business_chain, recommend_chain, retriever, escalation_chain

# โหลด
with st.spinner("กำลังโหลดฐานข้อมูล..."):
    business_chain, recommend_chain, retriever, escalation_chain = load_components()
st.success("พร้อมใช้งานแล้วค่ะ ✈️")

# Logic แนะนำแผนเบื้องต้น
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
# Chat UI
# =====================
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
                history = "\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in st.session_state.chat_history[-8:]
                )

                # === ตรวจจับ abuse ก่อน ===
                abuse_level = classify_abuse_level(user_input)
                if abuse_level != "NONE":
                    logger.info(f"Detected abuse - Level: {abuse_level} | Message: {user_input}")
                    response = escalation_chain.invoke({
                        "level": abuse_level,
                        "question": user_input
                    })
                    mode = f"ESCALATION_{abuse_level}"
                else:
                    # === ปกติ ===
                    recommend_keywords = ["แนะนำ", "ควรซื้อ", "แผนไหน", "เหมาะกับ", "ไป", "เที่ยว", "ญี่ปุ่น", "เกาหลี", "ยุโรป", "ต่างประเทศ"]
                    is_recommend = any(kw in user_input.lower() for kw in recommend_keywords)

                    trip_slots = extract_trip_slots(user_input)
                    recommend_info = simple_recommend(trip_slots)

                    with get_openai_callback() as cb:
                        if is_recommend or recommend_info["recommended_plans"]:
                            logger.info("Using RECOMMEND prompt")
                            mode = "RECOMMEND"
                            response = recommend_chain.invoke({
                                "trip_slots": json.dumps(trip_slots, ensure_ascii=False),
                                "recommended_plans_json": json.dumps(recommend_info["recommended_plans"], ensure_ascii=False),
                                "chat_history": history,
                                "question": user_input
                            })
                        else:
                            logger.info("Using BUSINESS prompt")
                            mode = "BUSINESS"
                            response = business_chain.invoke({
                                "question": user_input,
                                "chat_history": history,
                                "buy_link": "https://www.indara.co.th/products/travel-insurance/ta-allinone"
                            })

                        logger.info(f"Completed | Mode: {mode} | Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}")
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