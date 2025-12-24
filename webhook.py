# webhook.py – เวอร์ชันสมบูรณ์ แก้ error Runnable แล้ว (19 ธ.ค. 2025)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re

load_dotenv()
app = FastAPI()

# =====================
# Load Embedding & Vectorstore
# =====================
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
    collection_name="subacar_all"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# =====================
# Abuse Detection (เหมือน app.py)
# =====================
def classify_abuse_level(message: str) -> str:
    if not message or not message.strip():
        return "NONE"
    
    message_lower = message.lower().strip()
    
    # ดัก "บ้าง" ก่อน (คำถามสุภาพส่วนใหญ่)
    if 'บ้าง' in message_lower:
        if re.search(r'(เหี้ย|ควาย|สัส|เย็ด|มึง|กู|แม่ง|ไอ้สัตว์|อีสัตว์|ชิบหาย)', message_lower):
            return "MEDIUM"
        else:
            return "NONE"
    
    # HIGH: คุกคาม / ใส่ร้ายบริษัท
    threat_words = ['ฆ่า', 'ตาย', 'ขู่', 'ทำร้าย', 'ฟ้อง', 'แจ้งความ', 'ทำลาย', 'เผา', 'จัดการ', 'กรูจะ']
    slander_words = ['โกง', 'หลอก', 'ตุ๋น', 'ฉ้อโกง', 'scam', 'fraud', 'หลอกลวง', 'โจร']
    has_threat = any(word in message_lower for word in threat_words)
    has_slander = any(word in message_lower for word in slander_words)
    has_company = any(k in message_lower for k in ['subacar', 'ซุบะคาร์', 'suba car', 'บริษัท', 'พวกมึง', 'พวกนี้', 'suba'])
    
    if has_threat or (has_slander and has_company):
        return "HIGH"
    
    # MEDIUM: คำหยาบชัดเจน
    vulgar_words = [
        'เหี้ย', 'ไอ้เหี้ย', 'ควาย', 'สัส', 'เย็ด', 'มึง', 'กู', 'แม่ง', 'ชิบหาย',
        'ไอ้สัตว์', 'อีสัตว์', 'อีดอก', 'ส้นตีน', 'หน้าตัวเมีย', 'เหี้ยเอ้ย',
        'ไอ้', 'ไอ', 'เวร', 'สัตว์', 'เพี้ยน', 'หน้าตัว'
    ]
    strong_bully_words = [
        'โง่', 'ปัญญาอ่อน', 'งี่เง่า', 'หน้าเงิน', 'เลว', 'ห่วยแตก', 'ขี้ข้า'
    ]
    
    if any(word in message_lower for word in vulgar_words) or any(word in message_lower for word in strong_bully_words):
        return "MEDIUM"
    
    # LOW: น้ำเสียงไม่สุภาพเล็กน้อย
    mild_negative_words = ['ห่วย', 'แย่', 'ขี้เกียจ', 'ไร้สาระ', 'ด้อย', 'เสียดาย']
    excessive_exclamation = re.search(r'[!?]{6,}', message)
    rude_ending = re.search(r'\bวะ\b|\bเว้ย\b|\bเฮ้ย\b', message_lower)
    
    if any(word in message_lower for word in mild_negative_words) or excessive_exclamation or rude_ending:
        return "LOW"
    
    return "NONE"

# =====================
# Escalation Chain
# =====================
escalation_template = """
คุณคือระบบป้องกันความสุภาพของ Chatbot SUB A CAR

ระดับความรุนแรงที่ตรวจพบ: {level}
ข้อความผู้ใช้: {question}

ตอบตามระดับนี้เท่านั้น:

ถ้า LOW:
ขอความร่วมมือใช้ถ้อยคำที่สุภาพในการสนทนานะคะ 😊
หากมีคำถามเกี่ยวกับบริการ สามารถสอบถามใหม่ได้เลยค่ะ 🚗

ถ้า MEDIUM:
ขออภัยนะคะ ระบบไม่สามารถตอบข้อความที่มีถ้อยคำไม่เหมาะสมได้ค่ะ
ขอความร่วมมือใช้ภาษาที่สุภาพในการสนทนานะคะ
หากมีคำถามเกี่ยวกับบริการรถเช่าของ SUB A CAR สามารถสอบถามใหม่ได้เลยค่ะ 🚗

ถ้า HIGH:
ขออภัยค่ะ ระบบไม่สามารถดำเนินการสนทนาที่มีถ้อยคำไม่เหมาะสมต่อได้ค่ะ
หากต้องการความช่วยเหลือ สามารถติดต่อเจ้าหน้าที่ผ่าน Line @subacar หรือ Call Center โดยตรงได้นะคะ

ห้ามตอบข้อมูลบริการใน MEDIUM/HIGH
ห้ามตอบโต้หรือตำหนิ

ตอบเลย:"""

escalation_prompt = PromptTemplate.from_template(escalation_template)
escalation_chain = escalation_prompt | llm | StrOutputParser()

# =====================
# RAG Prompts
# =====================
rental_template = """
คุณคือพนักงาน SubaCar ที่น่ารัก เป็นกันเอง และเชี่ยวชาญด้านรถเช่า
ตอบภาษาไทยเท่านั้น

ใช้เฉพาะข้อมูลจากฐานข้อมูลด้านล่างนี้เท่านั้น ห้ามแต่งเพิ่ม

รูปแบบคำตอบ:
- ตอบสั้นมาก ใช้ bullet ไม่เกิน 5 ข้อ
- ความยาวรวมไม่เกิน 450 ตัวอักษร
- ใช้อิโมจิที่เหมาะสม 🚗✅
- เรียงราคาถูก → แพง
- บอกราคาชัดเจน + นโยบายยกเลิก (ถ้ามี)

ข้อมูลจากฐานข้อมูลจริง:
{context}

คำถามลูกค้า:
{question}

ถ้าไม่มีข้อมูลที่เกี่ยวข้องเลยจริง ๆ ให้ตอบ: "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"

หลังตอบข้อมูลแล้ว ชวนถามต่อ 1 ประโยคสั้น ๆ (เช่น ระยะเวลาใช้งาน, ประเภทรถที่สนใจ)

จบด้วยลิงก์นี้เสมอ:
https://www.subacar.co.th/package

ตอบเลย:"""

faq_template = """
คุณคือเจ้าหน้าที่ SubaCar ที่สุภาพ เป็นกันเอง น่ารัก
ตอบภาษาไทยเท่านั้น ตอบสั้น กระชับ อ่านง่าย

ใช้เฉพาะข้อมูลจาก FAQ ใน context เท่านั้น ห้ามแต่งเพิ่ม

รูปแบบคำตอบ:
- ใช้ภาษาง่าย เป็นมิตร
- เพิ่มอิโมจิที่เหมาะสม 🚗✅❓💬
- ถ้าเป็นขั้นตอน ใช้ลำดับข้อ + อิโมจิ
- ความยาวรวมไม่เกิน 400 ตัวอักษร

ข้อมูลจาก FAQ จริง:
{context}

คำถามลูกค้า:
{question}

ถ้าไม่มีข้อมูลที่เกี่ยวข้องเลยจริง ๆ ให้ตอบ: "ขอโทษค่ะ ข้อมูลนี้ยังไม่มีในระบบค่ะ"

หลังตอบแล้ว ชวนถามต่อ 1 ประโยคสั้น ๆ เกี่ยวกับหัวข้อนั้น

จบด้วยลิงก์นี้เสมอ:
https://www.subacar.co.th/faq

ตอบเลย:"""

rental_prompt = PromptTemplate.from_template(rental_template)
faq_prompt = PromptTemplate.from_template(faq_template)

def is_faq_question(question: str) -> bool:
    keywords = ["ขั้นตอน", "จอง", "ยกเลิก", "มัดจำ", "เอกสาร", "รับรถ", "คืนรถ", "ประกัน", "เงื่อนไข", "ชำระเงิน", "ติดต่อ", "สาขา"]
    return any(k in question for k in keywords)

# =====================
# Webhook Endpoint (แก้ error Runnable แล้ว)
# =====================
@app.post("/")
async def webhook(request: Request):
    try:
        data = await request.json()
        query = (
            data.get("text")
            or data.get("queryInput", {}).get("text", {}).get("text")
            or ""
        ).strip()

        if not query:
            text = "สวัสดีค่ะ 🚗 ยินดีต้อนรับสู่ SubaCar ค่ะ อยากเช่ารถแบบไหน หรือมีคำถามอะไรบอกมาได้เลยนะคะ 💬"
        else:
            # ตรวจคำไม่สุภาพก่อน
            abuse_level = classify_abuse_level(query)

            if abuse_level != "NONE":
                text = escalation_chain.invoke({
                    "level": abuse_level,
                    "question": query
                })
            else:
                # ดึง context
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                # เลือก prompt และลิงก์
                if is_faq_question(query):
                    prompt_to_use = faq_prompt
                    final_link = "https://www.subacar.co.th/faq"
                else:
                    prompt_to_use = rental_prompt
                    final_link = "https://www.subacar.co.th/package"

                # แก้ตรงนี้: invoke แยกขั้นตอนแทน pipe ทั้งหมด
                input_data = {"context": context, "question": query}
                formatted = prompt_to_use.invoke(input_data)
                llm_out = llm.invoke(formatted)
                text = StrOutputParser().invoke(llm_out)

                if final_link not in text:
                    text += f"\n\n{final_link}"

        return JSONResponse(content={
            "fulfillment_response": {
                "messages": [{"text": {"text": [text]}}]
            }
        })

    except Exception as e:
        print(f"Webhook Error: {e}")
        return JSONResponse(content={
            "fulfillment_response": {
                "messages": [{"text": {"text": ["ขอโทษค่ะ ระบบกำลังปรับปรุงอยู่ค่ะ ลองใหม่ในอีกสักครู่นะคะ 🚗"]}}]
            }
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)