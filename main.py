"""
WhatsPrep WhatsApp webhook.

Converts the CLI flow into a state machine. Each inbound WhatsApp message
advances one step. Run: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import re
import json
import httpx
import requests
from datetime import datetime

from fastapi import FastAPI, Request, Response, BackgroundTasks
from openai import OpenAI
from dotenv import load_dotenv

from student_registry import StudentRegistry

load_dotenv()
app = FastAPI()

# ---------------------------------------------------------------- config
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
GRAPH_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

registry = StudentRegistry(
    api_key=os.getenv("WHATSPREP_API_KEY"),
    base_url="https://latam.whatsprep.com/api",
)

DB_FILE = "students.json"

# phone -> {"step": str, "data": dict}
SESSIONS: dict[str, dict] = {}

LEVELS = ["Primary 1", "Primary 2", "Primary 3",
          "Primary 4", "Primary 5", "Primary 6"]


# ---------------------------------------------------------------- sending
async def send_message(to: str, text: str):
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.post(
            GRAPH_URL,
            headers={
                "Authorization": f"Bearer {ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text},
            },
        )
        if r.status_code != 200:
            print("Send failed:", r.status_code, r.text)


# ---------------------------------------------------------------- llm helpers
def call_llm(messages: list, max_tokens: int = 50) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=max_tokens, messages=messages
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return ""


def wants_to_exit(text: str) -> bool:
    r = call_llm([{"role": "user", "content": f"""Does this message CLEARLY mean the person wants to leave, quit, or end the conversation?

Message: "{text}"

Return ONLY "yes" or "no".
- Only "yes" if unambiguously about leaving (exit, quit, bye, stop, no thanks, i'm done)
- Typos or partial words that could be answers (e.g. "mat" for "math") return "no"
- When unsure, return "no"
"""}], max_tokens=5)
    return r.lower() == "yes"


def check_if_done(text: str) -> bool:
    r = call_llm([{"role": "user", "content": f"""Does this message mean the person wants to stop, end, or has nothing more to ask?

Message: "{text}"

Return ONLY "yes" or "no".
- "no" on its own means YES they are done
- Only return "no" if they are clearly asking for math or a topic
"""}], max_tokens=5)
    return r.lower() == "yes"


def validate_name(text: str) -> str:
    r = call_llm([{"role": "user", "content": f"""Extract only the child's name from: "{text}"
Return ONLY the name, properly capitalised. If none, return "Unknown"."""}])
    if not r or r == "Unknown" or not re.match(r"^[A-Za-z]+(?: [A-Za-z]+)*$", r):
        return "Unknown"
    return r


def validate_level(text: str) -> str:
    r = call_llm([{"role": "user", "content": f"""Extract the primary school level from: "{text}"
Return ONLY "Primary 1" through "Primary 6", or "Unknown".
"p5" -> Primary 5 | "primary 3" -> Primary 3 | "grade 4" -> Unknown"""}])
    return r if r in LEVELS else "Unknown"


def validate_gender(text: str) -> str:
    r = call_llm([{"role": "user", "content": f"""Extract gender from: "{text}"
Return ONLY "Male", "Female", or "Unknown".
"boy" -> Male | "my daughter" -> Female | "idk" -> Unknown"""}])
    return r if r in ["Male", "Female"] else "Unknown"


def validate_subject(text: str) -> str:
    r = call_llm([{"role": "user", "content": f"""Does this message indicate the person wants Math practice questions?

Message: "{text}"

Return ONLY "Math" if yes, or "Unknown" if no."""}])
    return "Math" if r == "Math" else "Unknown"


def validate_difficulty(text: str) -> str:
    r = call_llm([{"role": "user", "content": f"""Extract difficulty from: "{text}"
Return ONLY "Easy", "Medium", or "Hard". If unclear, "Medium"."""}])
    return r if r in ["Easy", "Medium", "Hard"] else "Medium"


def extract_student_identifier(text: str) -> tuple:
    r = call_llm([{"role": "user", "content": f"""Extract either a student name or student ID from: "{text}"
- Student ID (S0012, or plain number like 15): return ID|<id>
- Child's name: return NAME|<name>
- Neither: return UNKNOWN|UNKNOWN"""}], max_tokens=20)
    try:
        kind, value = r.strip().split("|", 1)
        return kind.strip().upper(), value.strip()
    except Exception:
        return "UNKNOWN", "UNKNOWN"


# ---------------------------------------------------------------- local db
def load_database() -> list:
    try:
        if not os.path.exists(DB_FILE):
            return []
        with open(DB_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def save_student_local(name, level, gender, phone, student_id):
    students = load_database()
    students.append({
        "student_id": student_id, "name": name, "primary_level": level,
        "gender": gender, "phone": phone,
        "joined_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(DB_FILE, "w") as f:
        json.dump(students, f, indent=4)


def check_existing_student_local(phone: str) -> list | None:
    matches = [s for s in load_database() if s.get("phone", "").lstrip("+65") == phone]
    return matches or None


def find_student_locally(kind: str, value: str) -> dict | None:
    for s in load_database():
        if kind == "NAME" and s.get("name", "").lower() == value.lower():
            return s
        if kind == "ID" and str(s.get("student_id", "")) == str(value):
            return s
    return None


# ---------------------------------------------------------------- platform api
def get_topics(student_id, subject: str) -> list | None:
    try:
        sid = int(student_id)
    except (ValueError, TypeError):
        print(f"Student not on platform: {student_id}")
        return None

    subject_map = {"Math": "Mathematics", "Science": "Science", "English": "English"}
    try:
        r = requests.post(
            "https://latam.whatsprep.com/api/topics",
            headers=registry.headers,
            json={"student_id": sid, "subject": subject_map.get(subject, subject)},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("data") or None
        print(f"Topics failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Topics error: {e}")
    return None


def get_subtopics(student_id, topic_id: str) -> list | None:
    try:
        sid, tid = int(student_id), int(topic_id)
    except (ValueError, TypeError):
        return None
    try:
        r = requests.post(
            "https://latam.whatsprep.com/api/topics",
            headers=registry.headers,
            json={"student_id": sid, "topic_id": tid},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("data") or None
    except Exception as e:
        print(f"Subtopics error: {e}")
    return None


def generate_worksheet_url(student_id, topic_id: str, difficulty: str) -> str | None:
    try:
        sid, tid = int(student_id), int(topic_id)
    except (ValueError, TypeError):
        return None
    try:
        r = requests.post(
            "https://latam.whatsprep.com/api/generate-assessment",
            headers=registry.headers,
            json={"topic_id": tid, "student_id": sid, "difficulty": difficulty},
            timeout=30,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            return d.get("assessment_url") or d.get("assessment_ur") or d.get("url")
        print(f"Generate failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Generate error: {e}")
    return None


# ---------------------------------------------------------------- flow
def session_for(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {"step": "start", "data": {}}
    return SESSIONS[phone]


def format_topics(topics: list) -> str:
    lines = [f"{i}. {t.get('topic_name', f'Topic {i}')}" for i, t in enumerate(topics, 1)]
    return "\n".join(lines)


def menu_prompt(name: str) -> str:
    return (f"How can I help you today, {name}?\n\n"
            f"Tell me what you'd like, e.g. \"I want Math questions\".")


def start_registration(s: dict) -> str:
    s["step"] = "reg_name"
    return "Let's register a new student.\n\nWhat is the student's name?"


def confirm_details_text(d: dict) -> str:
    return (f"Please confirm your child's details:\n\n"
            f"Name: {d['name']}\n"
            f"Level: {d['level']}\n"
            f"Gender: {d['gender']}\n"
            f"Phone: {d['phone']}\n\n"
            f"Is this correct? (yes/no)")


def do_register(s: dict) -> str:
    d = s["data"]
    new = registry.register_student(d["name"], d["level"], d["gender"], d["phone"])
    student_id = (new or {}).get(
        "student_id", f"S{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    save_student_local(d["name"], d["level"], d["gender"], d["phone"], student_id)
    d["student_id"] = student_id
    s["step"] = "menu"
    return (f"{d['name']} has been registered with WhatsPrep.\n"
            f"Student ID: {student_id}\n\n{menu_prompt(d['name'])}")


def load_and_show_topics(s: dict) -> str:
    d = s["data"]
    topics = get_topics(d.get("student_id"), d["subject"])
    if not topics:
        s["step"] = "menu"
        return "Sorry, I couldn't load the topics right now. Please try again shortly."
    d["topics"] = topics
    s["step"] = "topic"
    return (f"Here are the available topics for {d['name']}:\n\n"
            f"{format_topics(topics)}\n\n"
            f"Reply with the number(s), e.g. \"1\" or \"1,2,3\".")


def handle(phone: str, text: str) -> str:
    """One inbound message in, one reply out."""
    s = session_for(phone)
    d = s["data"]
    step = s["step"]

    # Global exit, valid at any step
    if step not in ("reg_name",) and wants_to_exit(text):
        name = d.get("name", "")
        SESSIONS.pop(phone, None)
        return f"Thank you for using WhatsPrep{', ' + name if name else ''}. Goodbye!"

    # ---------- start: identify the parent by their WhatsApp number
    if step == "start":
        local_phone = phone[2:] if phone.startswith("65") else phone
        d["phone"] = local_phone

        kind, value = extract_student_identifier(text)
        if kind in ("NAME", "ID"):
            student = find_student_locally(kind, value)
            if student:
                d["candidate"] = student
                s["step"] = "confirm_student"
                lvl = student.get("primary_level") or "Unknown level"
                return f"Is your child {student['name']} in {lvl}? (yes/no)"

        found = registry.check_student_exists(local_phone) \
            or check_existing_student_local(local_phone)

        if found and len(found) == 1:
            d["candidate"] = found[0]
            s["step"] = "confirm_student"
            lvl = found[0].get("primary_level") or "Unknown level"
            return (f"Welcome back to WhatsPrep!\n\n"
                    f"Is your child {found[0]['name']} in {lvl}? (yes/no)")

        if found and len(found) > 1:
            d["found"] = found
            s["step"] = "pick_student"
            lines = [f"{i}. {c['name']} — {c.get('primary_level') or 'Unknown level'}"
                     for i, c in enumerate(found, 1)]
            return ("We found more than one child under this number:\n\n"
                    + "\n".join(lines) + "\n\nWhich child is this for? Reply with the number.")

        return "Hi! Welcome to WhatsPrep.\n\n" + start_registration(s)

    # ---------- pick between siblings
    if step == "pick_student":
        try:
            i = int(text.strip()) - 1
            chosen = d["found"][i]
        except (ValueError, IndexError):
            return f"Please reply with a number between 1 and {len(d['found'])}."
        d["candidate"] = chosen
        s["step"] = "confirm_student"
        lvl = chosen.get("primary_level") or "Unknown level"
        return f"Is your child {chosen['name']} in {lvl}? (yes/no)"

    # ---------- confirm the matched student
    if step == "confirm_student":
        ans = text.strip().lower()
        if ans in ("yes", "y"):
            c = d["candidate"]
            d.update({
                "name": c["name"],
                "level": c.get("primary_level", ""),
                "gender": c.get("gender", ""),
                "student_id": c.get("student_id", ""),
            })
            s["step"] = "menu"
            return f"Great, let's get started!\n\n{menu_prompt(d['name'])}"
        if ans in ("no", "n"):
            return start_registration(s)
        return "Please reply yes or no."

    # ---------- registration
    if step == "reg_name":
        name = validate_name(text)
        if name == "Unknown":
            return "I couldn't catch a valid name. Please send just the name, e.g. \"Ayushi\"."
        d["name"] = name
        s["step"] = "reg_level"
        return "What is their schooling level? (e.g. P1 to P6)"

    if step == "reg_level":
        level = validate_level(text)
        if level == "Unknown":
            return "Please send a level between Primary 1 and Primary 6, e.g. \"P4\"."
        d["level"] = level
        s["step"] = "reg_gender"
        return "What is their gender? (boy or girl)"

    if step == "reg_gender":
        gender = validate_gender(text)
        if gender == "Unknown":
            return "Please send boy or girl."
        d["gender"] = gender
        s["step"] = "confirm_details"
        return confirm_details_text(d)

    if step == "confirm_details":
        ans = text.strip().lower()
        if ans in ("yes", "y"):
            return do_register(s)
        if ans in ("no", "n"):
            s["step"] = "fix_field"
            return ("Which field would you like to fix?\n\n"
                    "1. Name\n2. Level\n3. Gender\n\nReply with the number.")
        return "Please reply yes or no."

    if step == "fix_field":
        choice = text.strip()
        mapping = {"1": ("reg_name", "New name:"),
                   "2": ("reg_level", "New level (e.g. P3):"),
                   "3": ("reg_gender", "New gender (boy/girl):")}
        if choice not in mapping:
            return "Please reply with 1, 2, or 3."
        s["step"], prompt = mapping[choice]
        d["returning_to_confirm"] = True
        return prompt

    # ---------- main menu
    if step == "menu":
        if check_if_done(text):
            name = d.get("name", "")
            SESSIONS.pop(phone, None)
            return f"Thank you for using WhatsPrep, {name}. Goodbye!"

        subject = validate_subject(text)
        if subject == "Unknown":
            return ("We currently offer Math only.\n\n"
                    "Reply \"Math\" to see the available topics.")
        d["subject"] = subject
        return load_and_show_topics(s)

    # ---------- topic selection
    if step == "topic":
        topics = d["topics"]
        raw = re.split(r"[,\s]+", text.strip())
        indices, seen = [], set()
        for c in raw:
            if not c:
                continue
            try:
                i = int(c) - 1
            except ValueError:
                return f"\"{c}\" isn't a number. Please reply with numbers between 1 and {len(topics)}."
            if not (0 <= i < len(topics)):
                return f"\"{c}\" is out of range. Please pick between 1 and {len(topics)}."
            if i not in seen:
                seen.add(i)
                indices.append(i)

        if not indices:
            return "Please reply with a topic number."

        selected = [topics[i] for i in indices]
        d["topic_ids"] = [str(t.get("topic_id", "")) for t in selected]
        d["topic_names"] = [t.get("topic_name", "") for t in selected]

        names = "\n".join(f"• {n}" for n in d["topic_names"])

        if len(d["topic_ids"]) > 1:
            d["subtopic_id"] = None
            s["step"] = "difficulty"
            return (f"Selected:\n{names}\n\n"
                    f"Multiple topics selected, so we'll use the broad topics.\n\n"
                    f"What difficulty?\n1. Easy\n2. Medium\n3. Hard")

        s["step"] = "subtopic_choice"
        return (f"Selected:\n{names}\n\n"
                f"Would you like a specific subtopic, or the broad topic?\n\n"
                f"1. Show me subtopics\n2. Use the broad topic")

    # ---------- subtopic
    if step == "subtopic_choice":
        choice = text.strip()
        if choice == "2":
            d["subtopic_id"] = None
            s["step"] = "difficulty"
            return "What difficulty?\n1. Easy\n2. Medium\n3. Hard"
        if choice != "1":
            return "Please reply with 1 or 2."

        subs = get_subtopics(d.get("student_id"), d["topic_ids"][0])
        if not subs:
            d["subtopic_id"] = None
            s["step"] = "difficulty"
            return ("Couldn't load subtopics, so we'll use the broad topic.\n\n"
                    "What difficulty?\n1. Easy\n2. Medium\n3. Hard")

        d["subtopics"] = subs
        s["step"] = "subtopic_pick"
        lines = [f"{i}. {sub.get('subtopic_name', sub.get('name', f'Subtopic {i}'))}"
                 for i, sub in enumerate(subs, 1)]
        return (f"Subtopics under {d['topic_names'][0]}:\n\n"
                + "\n".join(lines) + "\n\nReply with the number.")

    if step == "subtopic_pick":
        subs = d["subtopics"]
        try:
            i = int(text.strip()) - 1
            chosen = subs[i]
        except (ValueError, IndexError):
            return f"Please reply with a number between 1 and {len(subs)}."
        d["subtopic_id"] = chosen.get("subtopic_id", chosen.get("id", ""))
        d["subtopic_name"] = chosen.get("subtopic_name", chosen.get("name", ""))
        s["step"] = "difficulty"
        return (f"Subtopic: {d['subtopic_name']}\n\n"
                f"What difficulty?\n1. Easy\n2. Medium\n3. Hard")

    # ---------- difficulty, then generate
    if step == "difficulty":
        mapping = {"1": "Easy", "2": "Medium", "3": "Hard"}
        difficulty = mapping.get(text.strip()) or validate_difficulty(text)
        d["difficulty"] = difficulty

        ids = [d["subtopic_id"]] if d.get("subtopic_id") else d["topic_ids"]

        urls = []
        for tid in ids:
            url = generate_worksheet_url(d.get("student_id"), str(tid).strip(), difficulty)
            if url:
                urls.append(url)

        s["step"] = "menu"

        if not urls:
            return ("Sorry, worksheet generation failed. Please try again shortly.\n\n"
                    "Anything else I can help with?")

        if len(urls) == 1:
            return (f"Your worksheet is ready.\n\n{urls[0]}\n\n"
                    f"Anything else I can help with?")

        lines = "\n".join(f"Worksheet {i}: {u}" for i, u in enumerate(urls, 1))
        return f"Your worksheets are ready.\n\n{lines}\n\nAnything else I can help with?"

    # ---------- fallback
    s["step"] = "menu"
    return menu_prompt(d.get("name", "there"))


# ---------------------------------------------------------------- webhook
@app.get("/")
async def verify(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=p.get("hub.challenge"), media_type="text/plain")
    return Response(status_code=403)


def process(phone: str, text: str):
    """Runs in the background so we can ACK Meta immediately."""
    import asyncio
    try:
        reply = handle(phone, text)
    except Exception as e:
        print(f"Handler error: {e}")
        reply = "Something went wrong on our end. Please try again."
    asyncio.run(send_message(phone, reply))


@app.post("/")
async def receive(request: Request, background: BackgroundTasks):
    body = await request.json()

    try:
        value = body["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError):
        return {"status": "ignored"}

    # Delivery / read receipts for messages we sent
    for st in value.get("statuses", []):
        print(f"STATUS {st.get('status')} -> {st.get('recipient_id')} "
              f"({st.get('id')}) {st.get('errors', '')}")

    msgs = value.get("messages", [])
    if not msgs:
        return {"status": "ok"}

    msg = msgs[0]
    text = (msg.get("text") or {}).get("body", "").strip()
    if not text:
        return {"status": "no_text"}

    background.add_task(process, msg["from"], text)
    return {"status": "ok"}

    