"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.

นักศึกษาต้องเติม TODO ใน 4 จุดด้านล่างใน Session 2 Lab 1.3
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import gspread
import requests
from google.oauth2.service_account import Credentials

TZ_TH = timezone(timedelta(hours=7))  # UTC+7


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """TODO 1: ใช้ gspread เปิด Sheet ของตัวเอง แล้ว append_row ด้วย [timestamp, menu, qty, price, total]

    Returns dict {timestamp, menu, qty, price, total} ที่ append แล้ว
    Raises RuntimeError ถ้า credentials ไม่มี หรือ Sheet ไม่ accessible
    """
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.environ.get("SHEET_ID")
    if not creds_json:
        raise RuntimeError(
            "ไม่พบ env GOOGLE_SHEETS_CREDENTIALS (ตั้ง secret แล้ว restart Codespace หรือยัง)")
    if not sheet_id:
        raise RuntimeError("ไม่พบ env SHEET_ID (ก๊อปจาก URL ของ Google Sheet)")

    try:
        info = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"GOOGLE_SHEETS_CREDENTIALS ไม่ใช่ JSON ที่ถูกต้อง: {exc}") from exc

    total = qty * price
    timestamp = datetime.now(TZ_TH).isoformat(timespec="seconds")

    try:
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(sheet_id).sheet1
        ws.append_row([timestamp, menu, qty, price, total])
    except Exception as exc:
        raise RuntimeError(f"เข้าถึง Google Sheet ไม่ได้: {exc}") from exc

    return {
        "timestamp": timestamp,
        "menu": menu,
        "qty": qty,
        "price": price,
        "total": total,
    }


def send_notification(message: str) -> str:
    """TODO 2: ส่ง message ไปยัง Telegram bot (ใช้ TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
    หรือ LINE bot (ใช้ LINE_CHANNEL_TOKEN) เลือกตัวใดตัวหนึ่ง

    Returns: provider name ที่ใช้ ("telegram" หรือ "line")
    Raises RuntimeError ถ้า no credentials
    """
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    line_token = os.environ.get("LINE_CHANNEL_TOKEN")

    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        r = requests.post(
            url, json={"chat_id": tg_chat_id, "text": message}, timeout=10)
        r.raise_for_status()
        return "telegram"

    raise RuntimeError(
        "ไม่พบ credentials ของ bot — ตั้ง TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID หรือ LINE_CHANNEL_TOKEN"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนขวด")
    parser.add_argument("--price", type=float,
                        required=True, help="ราคาต่อขวด")
    args = parser.parse_args()

    try:
        # TODO 3: เรียก append_to_sheet แล้ว extract total
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        # TODO 4: เรียก send_notification ด้วย message ที่บอกยอดที่บันทึก
        provider = send_notification(
            f"🧾 บันทึก {args.menu} x{args.qty} = {total:g} บาท ({row['timestamp']})")
    except Exception as exc:
        print(
            f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว กรึณาตรวจสอบ: {exc}", file=sys.stderr)
        return 0

    print(
        f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total:g} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())
