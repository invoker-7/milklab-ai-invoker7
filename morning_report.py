"""MilkLab Morning Report (S2).

Usage:
    python morning_report.py --today --dry-run      # เทสกับข้อมูลที่เพิ่ง log วันนี้
    python morning_report.py --date 2026-07-14      # ระบุวันเอง
    python morning_report.py                        # default = เมื่อวาน (ใช้กับ cron ตอนเช้า)
    python morning_report.py --debug --dry-run      # ดูว่าอ่านได้กี่แถว มีวันที่อะไรบ้าง
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta

import gspread
import requests
from google.oauth2.service_account import Credentials

TZ_TH = timezone(timedelta(hours=7))

COLUMNS = ["timestamp", "menu", "qty", "price", "total"]
COL = {name: i for i, name in enumerate(COLUMNS)}


# ---------- helpers (pure) ----------

def to_number(value) -> float:
    """แปลงค่าจาก Sheet เป็นตัวเลข ทนทานต่อ ',' ' ' '฿' และช่องว่าง

    Sheet ที่ตั้ง format ตัวเลขไว้อาจคืนค่ามาเป็น '1,300' ซึ่ง float() จะพังทันที
    """
    if value is None:
        return 0.0
    s = str(value).strip().replace(",", "").replace("฿", "")
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def row_date(raw_ts: str) -> str:
    """ดึงส่วน 'YYYY-MM-DD' ออกจาก timestamp ไม่ว่าจะเก็บมาในรูปแบบไหน

    รองรับ:
        2026-07-14T20:13:45+07:00   (ISO ที่ sales_logger เขียน)
        2026-07-14 20:13:45         (Sheets แปลงเป็น datetime ของมันเอง)
        14/07/2026 20:13:45         (Sheets locale ไทย -> วัน/เดือน/ปี)
    """
    s = str(raw_ts).strip()

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)          # YYYY-MM-DD...
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)      # DD/MM/YYYY...
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    return ""


def is_header(row: list) -> bool:
    """แถวนี้เป็น header หรือไม่ (กันกรณีชีตไม่มี/มี header ไม่เหมือนกัน)"""
    if not row:
        return True
    first = str(row[0]).strip().lower()
    return first in {"timestamp", "วันที่", "เวลา", "date"} or row_date(row[0]) == ""


# ---------- I/O layer ----------

def _sheet_id() -> str:
    sid = os.environ.get("SHEET_ID") or os.environ.get("GOOGLE_SHEETS_ID")
    if not sid:
        raise RuntimeError("ไม่พบ env SHEET_ID (หรือ GOOGLE_SHEETS_ID)")
    return sid


def read_rows() -> list:
    """ดึงทุกแถวจาก Sheet แล้วตัด header ออกแบบฉลาด (ไม่ใช่ตัดแถวแรกทิ้งดื้อ ๆ)"""
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not creds_json:
        raise RuntimeError("ไม่พบ env GOOGLE_SHEETS_CREDENTIALS")
    try:
        info = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"GOOGLE_SHEETS_CREDENTIALS ไม่ใช่ JSON ที่ถูกต้อง: {exc}") from exc

    try:
        creds = Credentials.from_service_account_info(
            info, scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        ws = gspread.authorize(creds).open_by_key(_sheet_id()).sheet1
        values = ws.get_all_values()
    except Exception as exc:
        raise RuntimeError(f"อ่าน Google Sheet ไม่ได้: {exc}") from exc

    # เก็บเฉพาะแถวที่ timestamp แปลงเป็นวันที่ได้จริง -> header และแถวว่างหลุดออกเอง
    return [r for r in values if not is_header(r)]


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )
    r.raise_for_status()


# ---------- Pure functions ----------

def summarize_for_date(rows: list, date: str) -> dict:
    """สรุปยอดขายของวันที่ระบุ — ไม่อ่าน Sheet เอง จึง unit-test ได้ทันที"""
    matched = [r for r in rows if r and row_date(r[COL["timestamp"]]) == date]

    total_qty = 0
    total_baht = 0.0
    by_menu = {}

    for r in matched:
        menu = str(r[COL["menu"]]).strip() or "(ไม่ระบุเมนู)"
        qty = int(to_number(r[COL["qty"]]))
        price = to_number(r[COL["price"]])
        total = to_number(r[COL["total"]]) if len(r) > COL["total"] else 0.0
        if total == 0.0:                     # เผื่อคอลัมน์ total ว่าง -> คำนวณเอง
            total = qty * price

        total_qty += qty
        total_baht += total

        slot = by_menu.setdefault(menu, {"qty": 0, "baht": 0.0})
        slot["qty"] += qty
        slot["baht"] += total

    top_menu = max(
        by_menu, key=lambda m: by_menu[m]["baht"]) if by_menu else None

    return {
        "date": date,
        "order_count": len(matched),
        "total_qty": total_qty,
        "total_baht": total_baht,
        "by_menu": by_menu,
        "top_menu": top_menu,
    }


def available_dates(rows: list) -> list:
    """วันที่ทั้งหมดที่มีข้อมูลอยู่จริงในชีต — ใช้ตอน debug"""
    return sorted({row_date(r[COL["timestamp"]]) for r in rows if row_date(r[COL["timestamp"]])})


def format_report(summary: dict) -> str:
    if summary["order_count"] == 0:
        return f"รายงานยอดขาย {summary['date']}\n\nไม่มีรายการขายในวันนี้"

    lines = [
        f"รายงานยอดขาย {summary['date']}",
        "",
        f"จำนวนออร์เดอร์: {summary['order_count']} รายการ",
        f"ขายได้ทั้งหมด: {summary['total_qty']} ชิ้น",
        f"ยอดรวม: {summary['total_baht']:,.0f} บาท",
        "",
        "แยกตามเมนู:",
    ]
    for menu, v in sorted(summary["by_menu"].items(), key=lambda kv: -kv[1]["baht"]):
        lines.append(f"  - {menu} x{v['qty']} = {v['baht']:,.0f} บาท")
    if summary["top_menu"]:
        lines += ["", f"ขายดีสุด: {summary['top_menu']}"]
    return "\n".join(lines)


def today_th() -> str:
    return datetime.now(TZ_TH).strftime("%Y-%m-%d")


def yesterday_th() -> str:
    return (datetime.now(TZ_TH) - timedelta(days=1)).strftime("%Y-%m-%d")


# ---------- main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="MilkLab Morning Report")
    parser.add_argument("--date", default=None,
                        help="วันที่ต้องการสรุป YYYY-MM-DD")
    parser.add_argument("--today", action="store_true",
                        help="สรุปยอดของวันนี้ (ใช้ตอนเทสกับแถวที่เพิ่ง log)")
    parser.add_argument("--dry-run", action="store_true",
                        help="พิมพ์รายงานออกหน้าจอ ไม่ส่ง Telegram")
    parser.add_argument("--debug", action="store_true",
                        help="แสดงจำนวนแถวที่อ่านได้ และวันที่ที่มีข้อมูลอยู่จริง")
    args = parser.parse_args()

    if args.date:
        date = args.date
    elif args.today:
        date = today_th()
    else:
        date = yesterday_th()

    try:
        rows = read_rows()
    except Exception as exc:
        print(f"[ERROR] อ่าน Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS / SHEET_ID "
              "และ share Sheet ให้ service account email", file=sys.stderr)
        return 1

    if args.debug:
        print(f"[DEBUG] อ่านข้อมูลได้ {len(rows)} แถว")
        print(
            f"[DEBUG] วันที่ที่มีข้อมูลในชีต: {available_dates(rows) or '(ไม่มีเลย)'}")
        print(f"[DEBUG] กำลังสรุปของวันที่: {date}")
        if rows:
            print(f"[DEBUG] ตัวอย่างแถวแรก: {rows[0]}")

    summary = summarize_for_date(rows, date)

    # เตือนให้ชัด แทนที่จะส่งรายงานว่างออกไปเงียบ ๆ
    if summary["order_count"] == 0 and rows:
        print(f"[WARN] ไม่พบรายการของวันที่ {date} "
              f"(ในชีตมีข้อมูลของวันที่: {', '.join(available_dates(rows))})", file=sys.stderr)
        print(
            "[HINT] ถ้าเพิ่งรัน sales_logger ไปเมื่อกี้ ให้ลอง --today", file=sys.stderr)

    report = format_report(summary)

    if args.dry_run:
        print("[DRY-RUN] ไม่ส่ง Telegram — ข้อความที่จะส่งคือ:\n")
        print(report)
        return 0

    try:
        send_telegram(report)
    except Exception as exc:
        print(f"[ERROR] ส่ง Telegram ล้มเหลว: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] ส่งรายงานวันที่ {date} แล้ว "
          f"({summary['order_count']} ออร์เดอร์ / {summary['total_baht']:,.0f} บาท)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
