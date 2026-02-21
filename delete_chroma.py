# delete_chroma.py - ลบฐานข้อมูล Chroma ทั้งหมด หรือเฉพาะ collection
import os
import shutil
from pathlib import Path

def delete_chroma_db():
    # กำหนด path ของ Chroma DB (เดียวกับใน app.py)
    base_dir = Path(__file__).resolve().parent
    chroma_dir = base_dir / "chroma_db_travel"

    if not chroma_dir.exists():
        print(f"ไม่พบโฟลเดอร์ Chroma DB ที่: {chroma_dir}")
        print("ไม่ต้องลบอะไรแล้วจบเลยครับ 😄")
        return

    print(f"พบโฟลเดอร์ Chroma DB ที่: {chroma_dir}")
    print("กำลังจะลบฐานข้อมูลทั้งหมด...")

    confirm = input("แน่ใจหรือไม่ว่าต้องการลบทั้งหมด? (พิมพ์ 'ใช่' เพื่อยืนยัน): ").strip()

    if confirm.lower() != "ใช่":
        print("ยกเลิกการลบเรียบร้อยครับ ไม่มีอะไรเปลี่ยนแปลง")
        return

    try:
        # วิธีที่ 1: ลบทั้งโฟลเดอร์ (แนะนำที่สุด ถ้าต้องการเริ่มใหม่ทั้งหมด)
        shutil.rmtree(chroma_dir)
        print(f"ลบโฟลเดอร์ทั้งหมดเรียบร้อย: {chroma_dir}")
        print("ฐานข้อมูล Chroma ถูกลบสมบูรณ์แล้ว")
        print("ครั้งต่อไปที่รัน app.py หรือ ingest.py จะสร้างฐานข้อมูลใหม่ให้อัตโนมัติ")

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการลบ: {e}")
        print("ลองปิดโปรแกรม/terminal ที่กำลังใช้ Chroma อยู่ แล้วรันใหม่ครับ")

if __name__ == "__main__":
    print("=== คำสั่งลบฐานข้อมูล Chroma สำหรับ INSURE Travel All in One ===\n")
    delete_chroma_db()
    print("\nจบการทำงาน")