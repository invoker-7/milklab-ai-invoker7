"""ดอทเอพอกะเทิน Caption Generator (S1).

Usage:
    python caption_generator.py

Reads GOOGLE_API_KEY from env. Generates a Thai caption for a Dota 2 skin listing.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของร้าน "ดอทเอพอกะเทิน" ร้านขายสกิน Dota 2

จงเขียนแคปชั่นภาษาไทย 2 ถึง 3 ประโยคโปรโมตสกิน: {menu}

เงื่อนไข:
- โทนสนุก ใช้คำง่าย ใส่ emoji ได้
- ต้องมี call-to-action ปิดท้าย เช่น สั่งเลย หรือ ทักแชท
- ห้ามใช้ em dash
"""


def generate_caption(menu: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for the given skin listing."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT_TEMPLATE.format(menu=menu),
    )
    return response.text or ""


def main() -> int:
    load_dotenv()
    menu = input("สกินที่จะโปรโมต: ").strip()
    if not menu:
        print("กรุณาใส่ชื่อสกิน")
        return 1
    caption = generate_caption(menu)
    print()
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
