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
from dotenv import load_dotenv
import re  # เพิ่มเข้ามาเพื่อช่วยตรวจ pattern

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
# Abuse Detection (เพิ่มใหม่)
# =====================
import re

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
        # ยกเว้นเฉพาะกรณีที่มีคำหยาบรุนแรงนำหน้าชัดเจน เช่น "รถเหี้ยบ้างไหม", "ไอ้ควายบ้าง"
        if re.search(r'(เหี้ย|ควาย|สัส|เย็ด|มึง|กู|แม่ง|ไอ้สัตว์|อีสัตว์|ชิบหาย)', message_lower):
            return "MEDIUM"
        else:
            # เช่น "มีรถบ้างไหมครับ", "มีโปรอะไรบ้างคะ", "มีสาขาแถวนี้บ้างไหม"
            return "NONE"
    
    # === HIGH: คุกคามหรือใส่ร้ายบริษัท ===
    threat_words = ['ฆ่า', 'ตาย', 'ขู่', 'ทำร้าย', 'ฟ้อง', 'แจ้งความ', 'ทำลาย', 'เผา', 'จัดการ', 'กรูจะ']
    slander_words = ['โกง', 'หลอก', 'ตุ๋น', 'ฉ้อโกง', 'scam', 'fraud', 'หลอกลวง', 'โจร']
    has_threat = any(word in message_lower for word in threat_words)
    has_slander = any(word in message_lower for word in slander_words)
    has_company = any(k in message_lower for k in ['subacar', 'ซุบะคาร์', 'suba car', 'บริษัท', 'พวกมึง', 'พวกนี้', 'suba'])
    
    if has_threat or (has_slander and has_company):
        return "HIGH"
    
    # === MEDIUM: คำหยาบหรือด่าชัดเจน (ลบ "บ้า" ออกแล้ว) ===
    vulgar_words = [
        'เหี้ย', 'ไอ้เหี้ย', 'ควาย', 'สัส', 'เย็ด', 'มึง', 'กู', 'แม่ง', 'ชิบหาย',
        'ไอ้สัตว์', 'อีสัตว์', 'อีดอก', 'ส้นตีน', 'หน้าตัวเมีย', 'เหี้ยเอ้ย',
        'ไอ้', 'ไอ', 'เวร', 'สัตว์', 'เพี้ยน', 'หน้าตัว'
        # ไม่มี "บ้า" อีกต่อไป
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
    excessive_exclamation = re.search(r'[!?]{6,}', message)  # 6 ตัวขึ้นไป
    rude_ending = re.search(r'\bวะ\b|\bเว้ย\b|\bเฮ้ย\b', message_lower)
    
    if any(word in message_lower for word in mild_negative_words) or excessive_exclamation or rude_ending:
        return "LOW"
    
    # ถ้าไม่เข้าหมวดไหนเลย → สุภาพ
    return "NONE"

# =====================
# Escalation Prompt (เพิ่มใหม่)
# =====================
fallback_escalation_template = """
คุณคือระบบป้องกันความสุภาพของ Chatbot SUB A CAR
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
หากมีคำถามเกี่ยวกับบริการรถเช่าของ SUB A CAR สามารถสอบถามใหม่ได้เลยค่ะ 🚗

ถ้า {level} = MEDIUM
→ ตั้งขอบเขตชัดเจน ห้ามตอบเนื้อหาคำถาม
ต้องตอบข้อความนี้เท่านั้น (ปรับได้เล็กน้อย แต่โครงสร้างเดียวกัน):
ขออภัยนะคะ ระบบไม่สามารถตอบข้อความที่มีถ้อยคำไม่เหมาะสมได้ค่ะ
ขอความร่วมมือใช้ภาษาที่สุภาพในการสนทนานะคะ
หากมีคำถามเกี่ยวกับบริการรถเช่าของ SUB A CAR สามารถสอบถามใหม่ได้เลยค่ะ 🚗

ถ้า {level} = HIGH
→ ยุติการสนทนาทันที ห้ามชวนคุยต่อ
ต้องตอบข้อความนี้เท่านั้น (ปรับได้เล็กน้อย แต่โครงสร้างเดียวกัน):
ขออภัยค่ะ ระบบไม่สามารถดำเนินการสนทนาที่มีถ้อยคำไม่เหมาะสมต่อได้ค่ะ
หากต้องการความช่วยเหลือ สามารถติดต่อเจ้าหน้าที่ผ่าน Line @subacar หรือ Call Center โดยตรงได้นะคะ

────────────────────────
ข้อห้ามเด็ดขาด:
- ห้ามตอบว่า "เข้าใจแล้วค่ะ" หรือ "ยินดีช่วยเสมอ" ในระดับ MEDIUM/HIGH
- ห้ามเปิดช่องให้ผู้ใช้ต่อความไม่สุภาพ
- ห้ามให้ข้อมูลบริการในระดับ MEDIUM และ HIGH

ตอบเลยตามระดับที่ระบุใน {level} เท่านั้น:
"""

# =====================
# Original functions (ไม่เปลี่ยนแปลง)
# =====================
def detect_answer_source(response_text: str, context_text: str) -> str:
    """
    ตรวจสอบแบบ heuristic ว่าคำตอบอ้างอิงจาก context หรือไม่
    """
    if not context_text or not response_text:
        return "unknown"

    # ตัด context ให้สั้นกัน token เยอะ
    context_snippet = context_text[:2000]

    # นับคำที่ซ้ำกันแบบง่าย
    common_terms = set(response_text.split()) & set(context_snippet.split())

    if len(common_terms) >= 3:
        return "from_context"
    else:
        return "ai_generated"

def get_context_text(question: str, retriever) -> str:
    """
    ดึง context text จาก retriever โดยตรง
    ใช้สำหรับ logging / QA เท่านั้น
    """
    try:
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join(d.page_content for d in docs)
    except Exception:
        return ""

def is_faq_question(question: str) -> bool:
    faq_keywords = [
        "ขั้นตอน", "จอง", "ยกเลิก", "มัดจำ", "เอกสาร",
        "รับรถ", "คืนรถ", "ประกัน", "เงื่อนไข",
        "ชำระเงิน", "ติดต่อ", "สาขา"
    ]
    return any(k in question for k in faq_keywords)

# =====================
# Streamlit Config
# =====================
st.set_page_config(
    page_title="SubaCar Bot",
    page_icon="🚗",
    layout="centered"
)

st.title("SubaCar Bot – รถเช่าราคาดีที่สุดในไทย")
st.caption("รถไฟฟ้า | ไฮบริด | SUV | Alphard | ถามเรื่องรถ หรือการจอง ยกเลิก ติดต่อ ก็ได้นะคะ 💬")

# =====================
# Session State
# =====================
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0.0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("สถิติการใช้ OpenAI (สะสม)")
    st.write(f"Total Tokens: {st.session_state.total_tokens:,}")
    st.write(f"ค่าใช้จ่ายประมาณ: ${st.session_state.total_cost:.6f}")

    if st.button("ล้างประวัติการสนทนา"):
        st.session_state.chat_history = []
        st.success("ล้างประวัติเรียบร้อยแล้วค่ะ")

# =====================
# Load RAG Chain (เพิ่ม escalation_chain)
# =====================
@st.cache_resource
def load_rag_chain():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding,
        collection_name="subacar_all"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )

    # --- Prompt เดิมทั้งหมด (ไม่เปลี่ยน) ---
    template = """
คุณคือพนักงาน SubaCar ที่น่ารัก เป็นกันเอง และเชี่ยวชาญด้านรถเช่า
ตอบภาษาไทยเท่านั้น

ใช้เฉพาะข้อมูลจากฐานข้อมูลด้านล่างนี้เท่านั้น
ห้ามแต่งข้อมูลเพิ่ม ห้ามอนุมาน ห้ามคำนวณราคาเองเด็ดขาด

สำคัญมาก:
ก่อนสรุปว่า "ไม่มีข้อมูล" ให้ตรวจสอบข้อมูลใน context อย่างละเอียด
- พิจารณาคำพ้อง ความหมายใกล้เคียง และข้อมูลที่เกี่ยวข้องทางอ้อม
- หาก context มีข้อมูลบางส่วนที่ตอบคำถามได้ ให้ตอบจากข้อมูลนั้น
- ห้ามตอบว่าไม่มีข้อมูล หาก context มีข้อมูลที่เกี่ยวข้องแม้เพียงบางส่วน

ให้ตอบว่า "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"
เฉพาะกรณีที่ context ไม่มีข้อมูลที่เกี่ยวข้องกับคำถามเลยจริง ๆ เท่านั้น

รูปแบบคำตอบ:
- ตอบสั้นมาก ใช้ bullet ไม่เกิน 5 ข้อ
- ความยาวรวมไม่เกิน 450 ตัวอักษร
- สามารถใช้อักษรย่อหรืออิโมจิที่ตรงกับข้อความได้
- เรียงราคาจากถูก → แพง (ถ้าเป็นเรื่องรถหรือแพ็กเกจ)
- หาก context มีราคาเพียง 1 ค่า หรือไม่สามารถเปรียบเทียบราคาได้ ให้แสดงราคาเท่าที่มี
- บอกราคาชัดเจนตามระยะเช่าที่มีใน context เท่านั้น
- บอกนโยบายยกเลิกทุกครั้ง (ถ้ามีข้อมูลใน context)

ข้อมูลจากฐานข้อมูลจริง:
{context}

ประวัติการสนทนาก่อนหน้า:
{chat_history}

คำถามลูกค้า (ข้อความล่าสุด):
{question}

แนวทางการตอบ (ทำตามลำดับ – เป็นกระบวนการภายใน):
1) ประเมินประเภทคำถาม:
- รุ่นรถ
- ราคา
- แพ็กเกจ
- เปรียบเทียบหลายรุ่น
- ภาพรวม / หาข้อมูล
- สถานที่ / จังหวัด (Location-only input)
- อื่น ๆ

────────────────────────
กรณี "สถานที่ / จังหวัด" เท่านั้น:
- หากข้อความผู้ใช้เป็นเพียงชื่อจังหวัด / สถานที่
- ไม่ถือว่าเป็นคำถามข้อมูล
- ห้ามตอบว่า "ไม่มีข้อมูล"
- ห้ามกล่าวถึงสาขา ความพร้อมให้บริการ หรือราคาในพื้นที่นั้น
- ให้ตอบเชิงรับรู้ความสนใจ + เชื่อมโยงการใช้งานรถเช่าแบบกว้าง
- ปิดท้ายด้วยคำถามใช้งาน 1 ข้อ (ตามกติกาด้านล่าง)
────────────────────────

2) รุ่นรถ:
- ระบุประเภทรถที่พบ (Sedan / SUV / EV / Hybrid)
- อธิบายลักษณะการใช้งานโดยรวม
- ห้ามลงสเปกเชิงเทคนิค

3) ราคา:
- แสดงเฉพาะราคาที่พบใน context
- หากมีหลายเงื่อนไข ให้แจ้งว่าเป็นราคาโดยประมาณ
- หากไม่ชัดเจน ให้หลีกเลี่ยงตัวเลข

4) แพ็กเกจ:
- อธิบายประเภทแพ็กเกจ (รายวัน / รายสัปดาห์ / รายเดือน)
- เน้นสิ่งที่รวมอยู่ในแพ็กเกจ

5) เปรียบเทียบหลายรุ่น:
- เปรียบเทียบเฉพาะรุ่น/ประเภทที่พบใน context
- ไม่เกิน 3 รุ่น
- เปรียบเทียบเชิงการใช้งาน
- ห้ามตัดสินว่ารุ่นใดดีกว่า
- หลีกเลี่ยงการเปรียบเทียบราคา หากข้อมูลไม่ครบ

6) คำถามภาพรวม:
- สรุปเป็นกลุ่มรถหรือประเภท
- ยกตัวอย่างรุ่นที่พบ
- ห้ามตอบว่าไม่มีข้อมูล

7) ตอบว่า "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"
เฉพาะกรณีที่ context ไม่เกี่ยวข้องเลยจริง ๆ เท่านั้น

หลังจากตอบข้อมูลหลักแล้ว:
- สร้างคำถามปิดท้าย 1 ข้อ จาก:
  • ระยะเวลาการใช้งาน
  • ประเภทรถที่สนใจ
  • ลักษณะการใช้งาน
- ตรวจสอบ chat_history ก่อน ห้ามถามซ้ำ
- ความยาวไม่เกิน 20–25 คำ
- โทนผู้ช่วย เป็นมิตร ไม่ขาย

จบคำตอบด้วยลิงก์นี้เสมอ:
https://www.subacar.co.th/package

ตอบเลย:"""  # (ใส่ template เดิมทั้งหมดของคุณที่นี่)
    prompt = PromptTemplate.from_template(template)

    faq_template = """
คุณคือเจ้าหน้าที่ SubaCar ที่สุภาพ เป็นกันเอง น่ารัก และตอบคำถามลูกค้าอย่างชัดเจน
ตอบภาษาไทยเท่านั้น ตอบสั้น กระชับ อ่านง่าย สบายตา

ใช้เฉพาะข้อมูลจาก FAQ ใน context เท่านั้น
ห้ามแต่งข้อมูล ห้ามเดา ห้ามอนุมาน หรือสรุปเกินที่มีใน FAQ เด็ดขาด

สำคัญมาก:
ก่อนบอกว่าไม่มีข้อมูล ให้ตรวจสอบ context อย่างละเอียด
- พิจารณาคำพ้อง ความหมายใกล้เคียง และหัวข้อที่เกี่ยวข้อง
- ถ้ามีข้อมูลบางส่วนที่ตอบได้ ให้ตอบจากข้อมูลนั้น
- ห้ามบอกว่าไม่มีข้อมูล หาก context มีส่วนที่เกี่ยวข้องแม้เพียงเล็กน้อย

ให้ตอบว่า "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ" 
เฉพาะกรณีที่ context ไม่มีข้อมูลที่เกี่ยวข้องเลยจริง ๆ

รูปแบบคำตอบ (สำคัญ):
- ใช้ภาษาง่าย สุภาพ เป็นมิตร
- เพิ่มอิโมจิที่เหมาะสม เพื่อให้อ่านสนุกและสบายตา (เช่น 🚗 ✅ ❓ 💬)
- ความยาวรวมไม่เกิน 400 ตัวอักษร
- ไม่ใส่ราคา เว้นแต่มีระบุชัดใน FAQ
- ไม่ขาย ไม่ชวนเช่า ไม่เสนอโปรโมชัน

กรณีคำถามทั่วไป:
- ตอบเป็นย่อหน้าเต็มประโยคสั้น ๆ หรือ bullet สั้น ๆ
- อ่านลื่นไหล กระชับ

กรณีคำถามเกี่ยวกับขั้นตอน วิธีการ กระบวนการ:
- ตอบเป็นลำดับข้อชัดเจน
- ใช้รูปแบบ:
  1. 💬 ขั้นตอนแรก...
  2. ✅ ขั้นตอนต่อไป...
- ไม่เกิน 5 ข้อ
- แต่ละข้อขึ้นบรรทัดใหม่

ข้อมูลจาก FAQ จริง:
{context}

ประวัติการสนทนาก่อนหน้า:
{chat_history}
(ใช้เพื่อเข้าใจบริบทเท่านั้น ห้ามนำข้อมูลใหม่จาก chat_history มาตอบ)

คำถามลูกค้า:
{question}

หลังจากตอบคำถามหลักแล้ว:
- เพิ่มข้อความปิดท้าย 1 ประโยคสั้น ๆ เพื่อชวนถามต่ออย่างสุภาพ
- ต้องเกี่ยวข้องกับหัวข้อคำถามล่าสุดเท่านั้น
- ใช้โทนเป็นมิตร ไม่ขาย ไม่ขอข้อมูลส่วนตัว
- ตัวอย่าง:
  • มีส่วนไหนอยากทราบเพิ่มเติมไหมคะ 💬
  • อยากให้ช่วยอธิบายขั้นตอนไหนเพิ่มเติมคะ ✅
  • สอบถามเรื่องนี้เพิ่มได้เลยนะคะ 😊

จบคำตอบด้วยลิงก์นี้เสมอ:
https://www.subacar.co.th/faq

ตอบเลย:"""  # (ใส่ faq_template เดิมทั้งหมดของคุณที่นี่)
    faq_prompt = PromptTemplate.from_template(faq_template)

    chain = (
        {
            "context": (
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
            ),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    faq_chain = (
        {
            "context": (
                RunnableLambda(lambda x: x["question"])
                | retriever
                | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
            ),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | faq_prompt
        | llm
        | StrOutputParser()
    )

    # --- เพิ่ม Escalation Chain ใหม่ ---
    escalation_prompt = PromptTemplate.from_template(fallback_escalation_template)
    escalation_chain = escalation_prompt | llm | StrOutputParser()

    return chain, faq_chain, retriever, escalation_chain

with st.spinner("กำลังโหลดฐานข้อมูลรถและ FAQ..."):
    chain, faq_chain, retriever, escalation_chain = load_rag_chain()

st.success("พร้อมแล้วค่ะ 😊")

# =====================
# Chat UI
# =====================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("อยากเช่ารถแบบไหน หรือมีคำถามอะไรคะ?"):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):
        with st.spinner("กำลังหาคำตอบให้..."):
            start_time = time.time()
            try:
                # --- ตรวจจับ abuse ก่อน ---
                abuse_level = classify_abuse_level(user_input)

                if abuse_level != "NONE":
                    logger.info(f"Detected inappropriate message - Level: {abuse_level}")
                    
                    response = escalation_chain.invoke({
                        "level": abuse_level,
                        "question": user_input
                    })
                    
                    # ไม่นับ token/cost และไม่ดึง context
                else:
                    # --- การทำงานเดิมทั้งหมด ---
                    with get_openai_callback() as cb:
                        history_text = "\n".join(
                            f"{m['role']}: {m['content']}"
                            for m in st.session_state.chat_history[-6:]
                        )
                        input_payload = {
                            "question": user_input,
                            "chat_history": history_text
                        }

                        if is_faq_question(user_input):
                            response = faq_chain.invoke(input_payload)
                            logger.info("Using FAQ prompt")
                        else:
                            response = chain.invoke(input_payload)
                            logger.info("Using RENTAL prompt")

                        context_text = get_context_text(user_input, retriever)
                        answer_source = detect_answer_source(response, context_text)
                        logger.info(f"Answer source check: {answer_source}")

                        st.session_state.total_tokens += cb.total_tokens
                        st.session_state.total_cost += cb.total_cost

                        logger.info(
                            f"Success | {time.time() - start_time:.2f}s | Tokens: {cb.total_tokens} | Cost: ${cb.total_cost:.6f}"
                        )

                end_time = time.time()
                elapsed_time = end_time - start_time

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

                st.markdown(response)
                st.caption(f"⏱️ ใช้เวลาตอบ {elapsed_time:.2f} วินาที")

            except Exception as e:
                logger.error(str(e))
                st.error("ขอโทษค่ะ เกิดข้อผิดพลาด ลองถามใหม่อีกครั้งนะคะ")