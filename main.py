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

# Set to True only once the platform /generate-assessment endpoint is confirmed
# to accept question_count and question_type. While False, the bot still picks
# sensible values internally but does not promise them to the parent.
SUPPORTS_QUESTION_OPTIONS = False

# phone -> {"step": str, "data": dict}
SESSIONS: dict[str, dict] = {}

# Message IDs already handled, so Meta's delivery retries don't send duplicate replies
PROCESSED_IDS: set[str] = set()

LEVELS = ["Primary 1", "Primary 2", "Primary 3",
          "Primary 4", "Primary 5", "Primary 6"]

# What a paper should look like for each level, so parents never have to say.
LEVEL_DEFAULTS = {
    "Primary 1": {"count": 10, "type": "MCQ"},
    "Primary 2": {"count": 10, "type": "MCQ"},
    "Primary 3": {"count": 15, "type": "MCQ"},
    "Primary 4": {"count": 15, "type": "Mixed"},
    "Primary 5": {"count": 20, "type": "Mixed"},
    "Primary 6": {"count": 25, "type": "Open-ended"},
}

# Phrases that suggest the parent wants a different child than the one we picked
SWITCH_HINTS = ("wrong", "different child", "another child", "not ", "switch",
                "other kid", "other child", "my other")


def defaults_for(level: str, difficulty: str) -> dict:
    """Sensible question count and type for a level, nudged by difficulty."""
    d = dict(LEVEL_DEFAULTS.get(level, {"count": 15, "type": "Mixed"}))
    if difficulty == "Hard":
        d["count"] = max(5, d["count"] - 5)
    elif difficulty == "Easy" and d["count"] < 25:
        d["count"] += 5
    return d


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


def is_finished(text: str, at_menu: bool = False) -> bool:
    """Merged replacement for wants_to_exit + check_if_done.

    One LLM call instead of two. The at_menu flag captures the only real
    difference between the old pair: at the menu a bare "no" means "nothing
    more", whereas mid-flow "no" is usually an answer to a question.
    """
    extra = ('- At this point a bare "no", "nope" or "nothing" means YES, they are finished.\n'
             if at_menu else
             '- A bare "no" here is probably answering a question, so return "no".\n')
    r = call_llm([{"role": "user", "content": f"""Does this message mean the person wants to stop, leave, or has nothing more to ask?

Message: "{text}"

Return ONLY "yes" or "no".
- "yes" only if clearly about leaving or being done (exit, quit, bye, stop, that's all, i'm done, no thanks)
{extra}- Typos or partial words that could be answers (e.g. "mat" for "math") return "no"
- If they are asking for practice or naming a topic, return "no"
- If they are asking about a different child, return "no"
- When unsure, return "no"
"""}], max_tokens=5)
    return r.lower().startswith("yes")


# ---------------------------------------------------------------- yes / no parsing
YES = {"yes", "y", "yup", "yeah", "yea", "ya", "yep", "yah", "sure", "ok", "okay",
       "okok", "correct", "right", "true", "confirm", "confirmed", "that's right",
       "thats right", "yes please", "correct la", "can", "go ahead", "please",
       "sounds good", "perfect", "great", "\U0001F44D", "\u2705","looks good", "lgtm", "go", "do it", "make it", "send it", "send",
       "alright", "cool", "fine", "good", "nice", "yes pls", "shoot",
       "let's go", "lets go", "start", "generate", "\U0001F64F",}

NO = {"no", "n", "nope", "nah", "naw", "not", "wrong", "incorrect", "false",
      "not really", "no lah", "cannot", "\u274C"}


def parse_yes_no(text: str):
    """Fast local check. Returns True / False / None (unclear). No API call."""
    if not text:
        return None
    t = text.strip().lower().strip(".!,?")
    if t in YES:
        return True
    if t in NO:
        return False
    return None


def ai_yes_no(text: str):
    """LLM fallback for phrasing the local check missed."""
    r = call_llm([
        {"role": "system", "content":
         "Classify a reply to a yes/no question. Answer with exactly one word: "
         "YES, NO, or UNCLEAR. Treat affirmations ('that's right', 'she is', "
         "'go ahead') as YES and denials ('not quite', 'wrong one') as NO."},
        {"role": "user", "content": text},
    ], max_tokens=5)
    a = (r or "").strip().upper()
    return True if a.startswith("YES") else False if a.startswith("NO") else None


def read_yes_no(text: str):
    """Local check first, LLM only when unclear."""
    ans = parse_yes_no(text)
    return ans if ans is not None else ai_yes_no(text)


# ---------------------------------------------------------------- request extraction
def extract_request(text: str, topics: list) -> dict:
    """Pull everything a parent stated in one message, in a single LLM call.

    Returns only the fields they actually indicated. Anything absent stays out,
    so the caller can fill it from level defaults instead of asking.
    """
    topic_list = "\n".join(
        f"{t.get('topic_id')}: {t.get('topic_name')}" for t in topics
    )
    r = call_llm([{"role": "user", "content": f"""A parent is asking for maths practice for their child.

Available topics:
{topic_list}

Their message: "{text}"

Return ONLY a JSON object, no markdown fences, no explanation:
{{"topic_id": <id from the list, or null>,
  "difficulty": "Easy" | "Medium" | "Hard" | null,
  "count": <number of questions, or null>,
  "type": "MCQ" | "Open-ended" | "Mixed" | null,
  "wants_subtopics": true | false
  "wants_subtopics": true | false,
  "objects": true | false}}}}

Rules:
- Only fill a field if the parent clearly indicated it. Use null otherwise.
- Match topics loosely: "decimals", "the fraction one", "fractions pls" should all match.
- "objects" is true only if they are pushing back, hesitating, or asking to change
  something. Approval, small talk, or anything neutral is false.
- A bare number like "2" means they picked topic number 2 from a list, not a count.
- "wants_subtopics" is true only if they asked to narrow down or see subtopics.
"""}], max_tokens=120)

    raw = (r or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.lower().startswith("json") else raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        print(f"extract_request could not parse: {raw!r}")
        return {}


# ---------------------------------------------------------------- validators
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
def get_topics(student_id, subject: str = "Math") -> list | None:
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


def generate_worksheet_url(student_id, topic_id: str, difficulty: str,
                           count: int | None = None,
                           qtype: str | None = None) -> str | None:
    try:
        sid, tid = int(student_id), int(topic_id)
    except (ValueError, TypeError):
        return None

    payload = {"topic_id": tid, "student_id": sid, "difficulty": difficulty}
    if SUPPORTS_QUESTION_OPTIONS:
        if count:
            payload["question_count"] = count
        if qtype:
            payload["question_type"] = qtype

    try:
        r = requests.post(
            "https://latam.whatsprep.com/api/generate-assessment",
            headers=registry.headers,
            json=payload,
            timeout=30,
        )
        if r.status_code == 200:
            d = r.json().get("data", {})
            return d.get("assessment_url") or d.get("assessment_ur") or d.get("url")
        print(f"Generate failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Generate error: {e}")
    return None


# ---------------------------------------------------------------- flow helpers
def session_for(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {"step": "start", "data": {}}
    return SESSIONS[phone]


def adopt_student(d: dict, c: dict) -> None:
    """Take a matched child as the active one. No confirmation needed."""
    d.update({
        "name": c["name"],
        "level": c.get("primary_level", ""),
        "gender": c.get("gender", ""),
        "student_id": c.get("student_id", ""),
    })
    # A different child means a fresh topic list and a fresh request
    for k in ("topics", "topic_id", "topic_name", "subtopic_id", "subtopic_name",
              "subtopics", "difficulty", "count", "type"):
        d.pop(k, None)


def format_topics(topics: list) -> str:
    return "\n".join(
        f"{i}. {t.get('topic_name', f'Topic {i}')}" for i, t in enumerate(topics, 1)
    )


def format_children(found: list) -> str:
    return "\n".join(
        f"{i}. {c['name']} — {c.get('primary_level') or 'Unknown level'}"
        for i, c in enumerate(found, 1)
    )


def menu_prompt(name: str) -> str:
    return (f"How can I help {name} today?\n\n"
            f"Just tell me what you'd like, e.g. \"easy fractions practice\".")


def start_registration(s: dict) -> str:
    s["step"] = "reg_name"
    return "Let's register a new student.\n\nWhat is the student's name?"


def confirm_details_text(d: dict) -> str:
    return (f"Please confirm your child's details:\n\n"
            f"Name: {d['name']}\n"
            f"Level: {d['level']}\n"
            f"Gender: {d['gender']}\n"
            f"Phone: {d['phone']}\n\n"
            f"Is this correct?")


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


def ensure_topics(s: dict) -> list | None:
    """Load the topic list once per child and cache it."""
    d = s["data"]
    if not d.get("topics"):
        d["topics"] = get_topics(d.get("student_id")) or []
    return d["topics"] or None


def topic_name_for(topics: list, topic_id) -> str:
    for t in topics:
        if str(t.get("topic_id")) == str(topic_id):
            return t.get("topic_name", "that topic")
    return "that topic"


def confirm_request_text(d: dict) -> str:
    """The single confirmation shown before anything is generated."""
    label = d.get("subtopic_name") or d.get("topic_name", "that topic")
    lines = [f"Okay great! Generating a practice on {label}, "
             f"{d['difficulty'].lower()} difficulty."]

    if SUPPORTS_QUESTION_OPTIONS:
        lines.append(f"\nI'll put together {d['count']} {d['type'].lower()} questions, "
                     f"which is what usually works well for "
                     f"{d.get('level', 'this level')}.")

    lines.append("\nLet me know if you want to change anything, for example "
                 "\"make it medium difficulty instead\" or \"show subtopics\". "
                 "Otherwise just say go ahead!")
    return "\n".join(lines)


def prepare_confirmation(s: dict) -> str:
    """Fill any gaps from level defaults, then ask for one confirmation."""
    d = s["data"]
    d.setdefault("difficulty", "Medium")
    picked = defaults_for(d.get("level", ""), d["difficulty"])
    d.setdefault("count", picked["count"])
    d.setdefault("type", picked["type"])
    s["step"] = "confirm_request"
    return confirm_request_text(d)


def apply_request(d: dict, req: dict, topics: list) -> None:
    """Merge whatever the parent stated into the pending request."""
    if req.get("topic_id"):
        d["topic_id"] = str(req["topic_id"])
        d["topic_name"] = topic_name_for(topics, req["topic_id"])
        d.pop("subtopic_id", None)
        d.pop("subtopic_name", None)
    if req.get("difficulty"):
        d["difficulty"] = req["difficulty"]
    if req.get("count"):
        d["count"] = int(req["count"])
    if req.get("type"):
        d["type"] = req["type"]


def wants_different_child(d: dict, text: str) -> bool:
    """Cheap local check so a mis-matched child can always be corrected."""
    if not d.get("found") or len(d["found"]) < 2:
        return False
    low = text.lower()
    return any(w in low for w in SWITCH_HINTS)


def offer_children(s: dict) -> str:
    d = s["data"]
    s["step"] = "pick_student"
    return "No problem! Which child is this for?\n\n" + format_children(d["found"])


def generate_and_reply(s: dict) -> str:
    d = s["data"]
    target = d.get("subtopic_id") or d.get("topic_id")
    url = generate_worksheet_url(
        d.get("student_id"), str(target).strip(), d["difficulty"],
        count=d.get("count"), qtype=d.get("type"),
    )

    # Clear the pending request but keep the child's profile for the next round
    for k in ("topic_id", "topic_name", "subtopic_id", "subtopic_name",
              "difficulty", "count", "type", "subtopics"):
        d.pop(k, None)
    s["step"] = "menu"

    if not url:
        return ("Sorry, I couldn't generate that worksheet just now. "
                "Please try again in a moment.")
    return (f"Here's the practice, ready to go!\n\n{url}\n\n"
            f"Anything else I can help with?")


# ---------------------------------------------------------------- main handler
def handle(phone: str, text: str) -> str:
    """One inbound message in, one reply out."""
    s = session_for(phone)
    d = s["data"]
    step = s["step"]

    # Global exit, valid at any step except when we're asking for a name
    if step not in ("reg_name",) and is_finished(text, at_menu=(step == "menu")):
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
                adopt_student(d, student)
                s["step"] = "menu"
                return f"Great, let's get started!\n\n{menu_prompt(d['name'])}"

        found = registry.check_student_exists(local_phone) \
            or check_existing_student_local(local_phone)

        if found and len(found) == 1:
            adopt_student(d, found[0])
            s["step"] = "menu"
            return f"Welcome back to WhatsPrep!\n\n{menu_prompt(d['name'])}"

        if found and len(found) > 1:
            d["found"] = found
            s["step"] = "pick_student"
            return ("Hi, welcome back! We found more than one child under this number:\n\n"
                    + format_children(found)
                    + "\n\nWhich child is this for? Reply with the number please.")

        return "Hi! Welcome to WhatsPrep.\n\n" + start_registration(s)

    # ---------- pick between siblings, then go straight to the menu
    if step == "pick_student":
        try:
            i = int(text.strip()) - 1
            chosen = d["found"][i]
        except (ValueError, IndexError):
            return f"Please reply with a number between 1 and {len(d['found'])}."
        adopt_student(d, chosen)
        s["step"] = "menu"
        return f"Great, let's get started!\n\n{menu_prompt(d['name'])}"

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
        ans = read_yes_no(text)
        if ans is True:
            return do_register(s)
        if ans is False:
            s["step"] = "fix_field"
            return ("Which field would you like to fix?\n\n"
                    "1. Name\n2. Level\n3. Gender\n\nReply with the number.")
        return "Sorry, I didn't quite catch that — is that a yes or a no?"

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

    # ---------- main menu: understand as much as possible from one message
    if step == "menu":
        # Escape hatch, since we no longer confirm which child was matched
        if wants_different_child(d, text):
            return offer_children(s)

        topics = ensure_topics(s)
        if not topics:
            return ("Sorry, I couldn't load the topics right now. "
                    "Please try again shortly.")

        req = extract_request(text, topics)
        apply_request(d, req, topics)

        if req.get("wants_subtopics") and d.get("topic_id"):
            return show_subtopics(s)

        if not d.get("topic_id"):
            s["step"] = "topic"
            return (f"Here's what {d.get('name', 'your child')} can practise:\n\n"
                    f"{format_topics(topics)}\n\n"
                    f"Just tell me which one, or say something like "
                    f"\"easy fractions\".")

        return prepare_confirmation(s)

    # ---------- topic selection, accepting a number or free text
    if step == "topic":
        if wants_different_child(d, text):
            return offer_children(s)

        topics = d.get("topics") or []

        raw = text.strip()
        if raw.isdigit():
            i = int(raw) - 1
            if not (0 <= i < len(topics)):
                return f"Please pick a number between 1 and {len(topics)}."
            d["topic_id"] = str(topics[i].get("topic_id", ""))
            d["topic_name"] = topics[i].get("topic_name", "")
        else:
            req = extract_request(text, topics)
            apply_request(d, req, topics)
            if req.get("wants_subtopics") and d.get("topic_id"):
                return show_subtopics(s)
            if not d.get("topic_id"):
                return ("I didn't catch which topic you meant. You can reply with "
                        f"a number from 1 to {len(topics)}, or name the topic.")

        return prepare_confirmation(s)

    # ---------- optional subtopic narrowing
    if step == "subtopic_pick":
        subs = d.get("subtopics") or []
        raw = text.strip()
        if raw.isdigit():
            i = int(raw) - 1
            if not (0 <= i < len(subs)):
                return f"Please pick a number between 1 and {len(subs)}."
            chosen = subs[i]
            d["subtopic_id"] = chosen.get("subtopic_id", chosen.get("id", ""))
            d["subtopic_name"] = chosen.get("subtopic_name", chosen.get("name", ""))
            return prepare_confirmation(s)
        # Anything else, treat as "just use the broad topic"
        return prepare_confirmation(s)

    # ---------- the single confirmation before generating
    if step == "confirm_request":
        if wants_different_child(d, text):
            return offer_children(s)

        if parse_yes_no(text) is True:      # fast path, no API call
            return generate_and_reply(s)

        topics = d.get("topics") or []
        req = extract_request(text, topics)

        if req.get("wants_subtopics"):
            return show_subtopics(s)

        if any(req.get(k) for k in ("topic_id", "difficulty", "count", "type")):
            apply_request(d, req, topics)
            return confirm_request_text(d)

        if parse_yes_no(text) is False or req.get("objects"):
            return ("No problem! What would you like to change? You can say things "
                    "like \"make it harder\", \"a different topic\", or "
                    "\"show subtopics\".")

        # Nothing to change and no pushback, so take it as a yes
        return generate_and_reply(s)

        topics = d.get("topics") or []
        req = extract_request(text, topics)

        if req.get("wants_subtopics"):
            return show_subtopics(s)

        if any(req.get(k) for k in ("topic_id", "difficulty", "count", "type")):
            apply_request(d, req, topics)
            return confirm_request_text(d)

        if ans is False:
            return ("No problem! What would you like to change? You can say things "
                    "like \"make it harder\", \"a different topic\", or "
                    "\"show subtopics\".")

        # Fall back to the slower classifier only when nothing else matched
        if ai_yes_no(text) is True:
            return generate_and_reply(s)

        return ("Sorry, I didn't quite catch that. Say \"go ahead\" and I'll make it, "
                "or tell me what to change.")

    # ---------- fallback
    s["step"] = "menu"
    return menu_prompt(d.get("name", "there"))


def show_subtopics(s: dict) -> str:
    """Only reached when a parent asks to narrow down, never forced on them."""
    d = s["data"]
    subs = get_subtopics(d.get("student_id"), d.get("topic_id"))
    if not subs:
        return ("I couldn't load subtopics for that one, so we'll use the broad "
                "topic.\n\n" + prepare_confirmation(s))
    d["subtopics"] = subs
    s["step"] = "subtopic_pick"
    lines = [f"{i}. {sub.get('subtopic_name', sub.get('name', f'Subtopic {i}'))}"
             for i, sub in enumerate(subs, 1)]
    return (f"Subtopics under {d.get('topic_name', 'that topic')}:\n\n"
            + "\n".join(lines)
            + "\n\nReply with a number, or say \"broad topic is fine\".")


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

    # Meta retries delivery if we're slow to ACK, which caused duplicate replies
    msg_id = msg.get("id")
    if msg_id in PROCESSED_IDS:
        return {"status": "duplicate"}
    PROCESSED_IDS.add(msg_id)
    if len(PROCESSED_IDS) > 1000:
        PROCESSED_IDS.clear()

    text = (msg.get("text") or {}).get("body", "").strip()
    if not text:
        return {"status": "no_text"}

    background.add_task(process, msg["from"], text)
    return {"status": "ok"}

