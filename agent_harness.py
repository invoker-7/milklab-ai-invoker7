"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from dotenv import load_dotenv
from google import genai


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]

TRACE_LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_trace.log")
TZ_TH = timezone(timedelta(hours=7))


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from raw model output, tolerating code fences or extra text."""
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(f"parse ไม่ได้: {raw}")
        data = json.loads(raw[start : end + 1])

    if not isinstance(data, dict):
        raise RuntimeError(f"ผลลัพธ์ต้องเป็น JSON object: {data!r}")
    return data


def _format_args_for_trace(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            rendered = value
        elif isinstance(value, float) and value.is_integer():
            rendered = str(int(value))
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "{" + ", ".join(parts) + "}"


def _append_trace_log(user_cmd: str, tool_call: dict[str, Any], tool_result: str) -> None:
    timestamp = datetime.now(TZ_TH).isoformat(timespec="seconds")
    llm_payload = json.dumps(tool_call, ensure_ascii=False, separators=(",", ": "))
    lines = [
        f"{timestamp} | user_input | {user_cmd}",
        f"{timestamp} | llm_response | {llm_payload}",
        f"{timestamp} | tool_result | {tool_result}",
    ]
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as trace_file:
        trace_file.write("\n".join(lines) + "\n")


def _today_th() -> str:
    return datetime.now(TZ_TH).strftime("%Y-%m-%d")


def _yesterday_th() -> str:
    return (datetime.now(TZ_TH) - timedelta(days=1)).strftime("%Y-%m-%d")


def _normalize_relative_query_date(cmd: str, tool_call: dict[str, Any]) -> dict[str, Any]:
    if tool_call.get("tool") != "query_sales":
        return tool_call

    cmd_text = cmd.strip()
    args = dict(tool_call.get("args") or {})

    if "วันนี้" in cmd_text:
        args["date"] = _today_th()
    elif "เมื่อวาน" in cmd_text:
        args["date"] = _yesterday_th()

    tool_call = dict(tool_call)
    tool_call["args"] = args
    return tool_call


def _validate_tool_call(tool_call: dict[str, Any]) -> None:
    tool = tool_call.get("tool")
    args = tool_call.get("args") or {}

    if tool == "log_sale":
        menu = str(args.get("menu", "")).strip()
        qty = int(args.get("qty", 0))
        price = float(args.get("price", 0))
        if not menu:
            raise RuntimeError("ValueError menu must not be empty")
        if qty <= 0:
            raise RuntimeError("ValueError quantity must be positive")
        if price <= 0:
            raise RuntimeError("ValueError price must be positive")
        return

    if tool == "query_sales":
        date = str(args.get("date", "")).strip()
        if not date:
            raise RuntimeError("ValueError date must not be empty")
        return

    if tool == "send_alert":
        message = str(args.get("message", "")).strip()
        if not message:
            raise RuntimeError("ValueError message must not be empty")
        return

    raise RuntimeError(f"unknown tool: {tool!r}")


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """TODO 1: ส่ง cmd ไป Gemini พร้อม TOOL_SCHEMA ขอให้ตอบเป็น JSON {tool, args}

    Returns dict {"tool": <name>, "args": <dict>}
    Raises RuntimeError ถ้า parse ไม่ได้
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    tool_names = ", ".join(tool["name"] for tool in TOOL_SCHEMA)
    prompt = f"""คุณคือ agent ที่ต้องเลือก tool ที่เหมาะที่สุดจากรายการนี้เท่านั้น:
{json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2)}

วันนี้คือ {_today_th()} เวลาไทย (UTC+7)

คำสั่งผู้ใช้: {cmd}

กติกา:
- ตอบกลับเป็น JSON object เท่านั้น
- รูปแบบต้องเป็น {{"tool": "<name>", "args": {{...}}}}
- tool ต้องเป็นหนึ่งใน: {tool_names}
- ห้ามใส่คำอธิบายอื่น นอกเหนือจาก JSON
"""

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    data = _extract_json_object(response.text or "")
    tool = data.get("tool")
    args = data.get("args")

    if tool not in {item["name"] for item in TOOL_SCHEMA}:
        raise RuntimeError(f"unknown tool: {tool!r}")
    if not isinstance(args, dict):
        raise RuntimeError(f"args ต้องเป็น object: {args!r}")

    return _normalize_relative_query_date(cmd, {"tool": tool, "args": args})


def dispatch_tool(tool_call: dict) -> str:
    """TODO 2: เรียก tool ตาม tool_call["tool"] ด้วย args จริง

    Returns: ข้อความสรุปผลที่ tool คืน
    """
    tool = tool_call.get("tool")
    args = tool_call.get("args") or {}

    _validate_tool_call(tool_call)

    if tool == "log_sale":
        from sales_logger import append_to_sheet, send_notification

        row = append_to_sheet(
            menu=str(args["menu"]),
            qty=int(args["qty"]),
            price=float(args["price"]),
        )
        provider = send_notification(
            f"🧾 บันทึก {row['menu']} x{row['qty']} = {row['total']:g} บาท ({row['timestamp']})"
        )
        return f"OK: row appended at {row['timestamp']}"

    if tool == "query_sales":
        from morning_report import format_report, read_rows, summarize_for_date

        date = str(args["date"])
        rows = read_rows()
        summary = summarize_for_date(rows, date)
        return format_report(summary)

    if tool == "send_alert":
        from sales_logger import send_notification

        provider = send_notification(str(args["message"]))
        return f"ส่งแจ้งเตือนผ่าน {provider} เรียบร้อย"

    raise RuntimeError(f"unknown tool: {tool!r}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")

    # TODO 3: เรียก parse_command then dispatch_tool then print trace ตาม format ใน session-2.md
    tool_call = {"tool": "parse_error", "args": {}}
    result = ""

    try:
        tool_call = parse_command(args.cmd)
        print(f"[LLM]  tool={tool_call['tool']} args={_format_args_for_trace(tool_call['args'])}")

        result = dispatch_tool(tool_call)
        print(f"[TOOL] {tool_call['tool']} {result}")
        if tool_call["tool"] == "log_sale":
            total = tool_call["args"]["qty"] * tool_call["args"]["price"]
            print(f"[USER] ←  บันทึกแล้วยอด {total:g} บาท")
        else:
            print(f"[USER] ← {result}")
        return 0
    except Exception as exc:
        result = f"ERROR: {exc}"
        print(f"[TOOL] {tool_call.get('tool', 'parse_error')} {result}")
        print(f"[USER] ← {result}", file=sys.stderr)
        return 1
    finally:
        _append_trace_log(args.cmd, tool_call, result)


if __name__ == "__main__":
    sys.exit(main())
