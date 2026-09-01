import os
import json
import hmac
import hashlib
import asyncio
import logging
import re
import tempfile
from collections import deque
from contextlib import suppress
from datetime import datetime

import httpx
import requests
from fastapi import FastAPI, Request, Response, BackgroundTasks
from openai import AsyncOpenAI
from dotenv import load_dotenv

from student_registry import StudentRegistry, normalize_level, LEVELS

load_dotenv()
app = FastAPI()

# ---------------------------------------------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("whatsprep")

# ---------------------------------------------------------------- config
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
APP_SECRET = os.getenv("META_APP_SECRET")          # see verify_signature
GRAPH_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
MEDIA_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/media"

registry = StudentRegistry(
    api_key=os.getenv("WHATSPREP_API_KEY"),
    base_url="https://latam.whatsprep.com/api",
)

DB_FILE = "students.json"

SUPPORTS_QUESTION_OPTIONS = False
SUPPORTS_PDF = True

REPORT_TEMPLATE_NAME = os.getenv("REPORT_TEMPLATE_NAME", "report_ready")
REPORT_TEMPLATE_LANG = os.getenv("REPORT_TEMPLATE_LANG", "en")
PLATFORM_CALLBACK_SECRET = os.getenv("PLATFORM_CALLBACK_SECRET")

LEVEL_DEFAULTS = {
    "Primary 1": {"count": 10, "type": "MCQ"},
    "Primary 2": {"count": 10, "type": "MCQ"},
    "Primary 3": {"count": 15, "type": "MCQ"},
    "Primary 4": {"count": 15, "type": "Mixed"},
    "Primary 5": {"count": 20, "type": "Mixed"},
    "Primary 6": {"count": 25, "type": "Open-ended"},
}

DIFFICULTY_EMOJI = {"Easy": "\U0001F7E2", "Medium": "\U0001F7E1", "Hard": "\U0001F534"}

MAX_HISTORY = 24        # messages kept per parent
MAX_TOOL_HOPS = 6       # safety valve on the agent loop
MAX_PROCESSED = 1000    # inbound message ids remembered for dedupe

# phone -> {"messages": [...], "child": {...} | None, "topics": [...] }
SESSIONS: dict[str, dict] = {}

PROCESSED_IDS: set[str] = set()
PROCESSED_ORDER: deque[str] = deque(maxlen=MAX_PROCESSED)


def already_processed(msg_id: str) -> bool:
    """True if this inbound id was already handled. Records it if not."""
    if not msg_id:
        return False
    if msg_id in PROCESSED_IDS:
        return True
    if len(PROCESSED_ORDER) == PROCESSED_ORDER.maxlen:
        PROCESSED_IDS.discard(PROCESSED_ORDER[0])
    PROCESSED_ORDER.append(msg_id)
    PROCESSED_IDS.add(msg_id)
    return False


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
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp", "to": to, "type": "text",
                  "text": {"body": text[:4096], "preview_url": True}},
        )
        if r.status_code != 200:
            log.warning("Send failed: %s %s", r.status_code, r.text)


async def send_document(to: str, filepath: str, filename: str,
                        caption: str = "") -> bool:
    """Send a PDF as a WhatsApp document: upload media, then send by id."""
    async with httpx.AsyncClient(timeout=60) as http:
        try:
            with open(filepath, "rb") as f:
                r = await http.post(
                    MEDIA_URL,
                    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                    data={"messaging_product": "whatsapp",
                          "type": "application/pdf"},
                    files={"file": (filename, f, "application/pdf")},
                )
        except OSError as e:
            log.error("Media file unreadable: %s", e)
            return False
        media_id = r.json().get("id") if r.status_code == 200 else None
        if not media_id:
            log.warning("Media upload failed: %s %s", r.status_code, r.text)
            return False

        r = await http.post(
            GRAPH_URL,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp", "to": to,
                  "type": "document",
                  "document": {"id": media_id, "filename": filename,
                               "caption": caption[:1024]}},
        )
        if r.status_code != 200:
            log.warning("Document send failed: %s %s", r.status_code, r.text)
            return False
        return True


async def send_report_template(to: str, student_name: str,
                               report_link: str) -> bool:
    payload = {
        "messaging_product": "whatsapp", "to": to, "type": "template",
        "template": {
            "name": REPORT_TEMPLATE_NAME,
            "language": {"code": REPORT_TEMPLATE_LANG},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "parameter_name": "student_name",
                     "text": str(student_name)[:512]},
                    {"type": "text", "parameter_name": "report_link",
                     "text": str(report_link)[:1024]},
                ],
            }],
        },
    }
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.post(
            GRAPH_URL,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                     "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code != 200:
            log.warning("Report template send failed: %s %s",
                        r.status_code, r.text)
            return False
        log.info("Report template sent to %s", to[-4:])
        return True


# ------------------------------------------------------- report watcher
REPORT_WATCHES: dict[int, asyncio.Task] = {}
REPORT_WATCH_INTERVAL = 300        # seconds between polls
REPORT_WATCH_LIFETIME = 6 * 3600   # give up after 6 hours


async def _watch_for_report(student_id: int, phone: str, student_name: str):
    try:
        baseline = await asyncio.to_thread(_get_student_reports, student_id)
        known = {r.get("quiz_id_explico") for r in (baseline or [])}
        waited = 0
        while waited < REPORT_WATCH_LIFETIME:
            await asyncio.sleep(REPORT_WATCH_INTERVAL)
            waited += REPORT_WATCH_INTERVAL
            reports = await asyncio.to_thread(_get_student_reports, student_id)
            if not reports:
                continue
            new = [r for r in reports
                   if r.get("quiz_id_explico") not in known]
            if new:
                latest = new[0]          # list is sorted newest first
                ok = await send_report_template(
                    phone, student_name, latest.get("report_link", ""))
                log.info("Report watcher fired for student %s (quiz %s): %s",
                         student_id, latest.get("quiz_id_explico"),
                         "sent" if ok else "SEND FAILED")
                return
        log.info("Report watcher expired for student %s", student_id)
    except asyncio.CancelledError:
        pass
    except Exception:
        log.exception("Report watcher crashed for student %s", student_id)
    finally:
        REPORT_WATCHES.pop(student_id, None)


def start_report_watch(student_id, phone: str, student_name: str):
    try:
        sid = int(student_id)
    except (ValueError, TypeError):
        return
    old = REPORT_WATCHES.get(sid)
    if old and not old.done():
        old.cancel()
    REPORT_WATCHES[sid] = asyncio.create_task(
        _watch_for_report(sid, phone, student_name))
    log.info("Report watcher started for student %s", sid)


# ---------------------------------------------------------------- local db
def load_database() -> list:
    try:
        if not os.path.exists(DB_FILE):
            return []
        with open(DB_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.error("Local db read error: %s", e)
        return []


def save_student_local(name, level, gender, phone, student_id):
    clean_name = re.sub(r"\bLast\b", "", str(name), flags=re.IGNORECASE).strip()
    students = load_database()
    students.append({
        "student_id": student_id, "name": clean_name, "primary_level": level,
        "gender": gender, "phone": phone,
        "joined_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(DB_FILE, "w") as f:
        json.dump(students, f, indent=4)


def check_existing_student_local(phone: str) -> list | None:
    matches = []
    for s in load_database():
        if s.get("phone", "").lstrip("+65") != phone:
            continue
        row = dict(s)
        row["primary_level"] = normalize_level(row.get("primary_level"))
        row["name"] = re.sub(r"\bLast\b", "", str(row.get("name", "")), flags=re.IGNORECASE).strip()
        matches.append(row)
    return matches or None


def usable_student_id(student_id) -> bool:
    try:
        int(student_id)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- platform api
def _get_topics(student_id, subject: str = "Math") -> list | None:
    try:
        sid = int(student_id)
    except (ValueError, TypeError):
        log.info("Student not on platform: %s", student_id)
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
            topics = r.json().get("data") or None
            log.info("Topics loaded: %s for student %s",
                     len(topics) if topics else 0, sid)
            return topics
        log.warning("Topics failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Topics error: %s", e)
    return None


def _get_subtopics(student_id, topic_id) -> list | None:
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
        log.warning("Subtopics failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Subtopics error: %s", e)
    return None


def _generate_worksheet_url(student_id, topic_ids, difficulty,
                            count=None, qtype=None, is_offline: int = 0) -> str | None:
    try:
        sid = int(student_id)
        tids = [int(t) for t in topic_ids]
    except (ValueError, TypeError):
        return None
    payload = {"topic_ids": tids, "student_id": sid,
               "difficulty": str(difficulty).lower(),
               "is_offline": is_offline}
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
            url = (d.get("assessment_url") or d.get("assessment_ur")
                   or d.get("pdf_url") or d.get("pdf_link")
                   or d.get("file_url") or d.get("url"))
            log.info("Generated worksheet for student %s topics %s: %s",
                     sid, tids, bool(url))
            return url
        log.warning("Generate failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Generate error: %s", e)
    return None


def _get_student_reports(student_id) -> list | None:
    try:
        sid = int(student_id)
    except (ValueError, TypeError):
        return None
    try:
        r = requests.post(
            "https://latam.whatsprep.com/api/student-reports",
            headers=registry.headers,
            json={"student_id": sid},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json().get("data") or []
            data = [d for d in data if d.get("report_link")]
            data.sort(key=lambda d: d.get("quiz_id_explico") or 0, reverse=True)
            log.info("Reports for student %s: %d", sid, len(data))
            return data
        log.warning("Reports failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Reports error: %s", e)
    return None


# ---------------------------------------------------------------- reply cleanup
_MD_LINK = re.compile(r"\[([^\]]+)\]\(\s*(https?://[^\s)]+)\s*\)")
_BARE_LINK = re.compile(r"https?://\S*whatsprep\.com\S*")


def clean_reply(text: str) -> str:
    if not text:
        return text
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_LINK.sub("", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- session
def session_for(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {"messages": [], "child": None, "topics": []}
    return SESSIONS[phone]


# ------------------------------------------------------- history integrity
def sanitize_history(messages: list) -> list:
    out, i, n = [], 0, len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")

        if role == "tool":
            i += 1
            continue

        if role == "assistant" and m.get("tool_calls"):
            ids = {tc.get("id") for tc in m["tool_calls"]}
            j, replies = i + 1, []
            while j < n and messages[j].get("role") == "tool":
                if messages[j].get("tool_call_id") in ids:
                    replies.append(messages[j])
                j += 1
            if {r.get("tool_call_id") for r in replies} == ids:
                out.append(m)
                out.extend(replies)
            i = j
            continue

        out.append(m)
        i += 1
    return out


def trim_history(messages: list, keep: int = MAX_HISTORY) -> list:
    return sanitize_history(messages[-keep:])


# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = """\
You are the WhatsPrep assistant. You help Singapore parents start Maths \
practice for their primary school child, over WhatsApp. Maths only. Never \
offer or discuss other subjects.

HOW THE PRODUCT WORKS
You do not send questions in the chat. You generate a practice set and send \
back a link the child opens. Never write maths questions yourself.

NEVER WRITE THE LINK YOURSELF
The system appends the practice link to the end of your own message \
automatically. If you also paste it, the parent gets it twice. Write one \
short line like "Here's Ayu's practice on Angles, medium difficulty" and \
stop. Do not write the URL, do not write markdown links like [text](url), \
and do not end with "here:" or "click this". WhatsApp shows markdown \
literally, so never use square brackets, asterisks or backticks anywhere.

ONE PAPER, ALL TOPICS
If a parent asks for several topics, pass them ALL in a single \
create_practice call as one topic_ids list. The platform combines them into \
one paper. Never call create_practice more than once for the same request, \
it produces several separate papers and several links.

PDF OPTION
If a parent asks for a printable paper, a PDF, or a hard copy, offer the \
PDF version. Before generating, tell them plainly: the PDF comes with an \
answer key, but it will NOT be marked by WhatsPrep and no \
progress report will be sent - marking and reports only happen for practice \
done through the link. If they confirm they want the PDF, call \
create_practice_pdf with the same topic_ids and difficulty rules as \
create_practice. The system sends the file itself; you just write one short \
line after, like "Sent! Answer key is included for you." Never call \
both create_practice and create_practice_pdf for the same request unless \
the parent explicitly asks for both.

REPORTS
When a parent asks how their child did, for scores, results, or the report, \
call get_report and share the latest report link in one short line. The \
system also notifies parents automatically when a fresh report is ready, so \
never promise to "send it later" yourself. If there are no reports yet, \
explain the report appears after their child finishes a practice set.

ASK AS FEW QUESTIONS AS POSSIBLE
Read everything the parent gave you in one message and act on it. \
"my girl is in p4, fractions are killing her" is enough to register or find \
the child, pick the topic, infer a difficulty and confirm. Only ask for \
something if it is genuinely missing and you cannot proceed without it. \
Never ask two questions in one message.

You never choose question count or question type. The system sets those \
from the child's level. Do not mention them unless the parent asks.

FLOW
1. Call find_children first, every new conversation. It looks up by the \
   parent's number automatically.
2. If exactly one child comes back, greet them by name and carry on. \
   If several, ask which one and call select_child. If none, you need the \
   child's name, level and gender, then call register_child.
3. Call list_topics before naming any topic. Never invent one.
4. Confirm in one short line, then call create_practice.
5. Send the link, then ask if there is anything else.

STYLE
Write like a helpful person texting. Short sentences, plain English, warm, \
no markdown, no bullet characters, no headings. You may use emoticons to convey tone. Many parents are not \
tech-savvy. One or two short messages, never a wall of text. When you list \
topics, number them so a parent can reply with a number.

OFF-TOPIC AND SMALL TALK
Reply like a person would: answer briefly and honestly, then offer what you \
can actually do. Never fall back on a canned "I didn't catch that" when the \
message was perfectly clear and simply not about practice.
  "what's the time right now?"
    -> "I can't check the time I'm afraid, but I can get some Maths practice
        going for Ayu whenever you're ready."
  "how are you?"
    -> "Doing well, thanks! What should Ayu work on today?"
  "can you help with science?"
    -> "Only Maths at the moment, sorry. Want me to set up some Maths
        practice instead?"
Never claim an ability you do not have. You cannot browse, check the time, \
send reminders, or see the child's homework. Say so plainly and move on.

ADDING ANOTHER CHILD
If a parent asks to add a student, they want to register another child. Ask \
for the name, level and gender in one message, then call register_child. Do \
not show them a topic list.

ASKING AGAIN
Only say you did not understand when you genuinely did not. If a parent \
says something unrelated, respond to what they actually said.

WHEN A TOOL RETURNS AN ERROR
Never invent a result and never carry on as if it worked. Each error carries \
a "message" telling you what to do; follow it. In particular, if a lookup or \
registration is unavailable, do NOT register anyone and do NOT offer to \
generate practice. Apologise in one short line and ask them to try again in a \
few minutes. If create_practice reports an unknown topic, use the valid list \
it returns and pick again rather than guessing.

SAFETY
Never ask for personal details beyond the child's first name, level and \
gender. Never ask for an address, school, birth date or contact details. \
If you are speaking with a child rather than a parent, stay on practice and \
never suggest keeping anything from their parent. If the conversation moves \
away from Maths practice, gently bring it back.
"""

# ---------------------------------------------------------------- tools
TOOLS = [
    {"type": "function", "function": {
        "name": "find_children",
        "description": ("Look up children already registered to this parent's "
                        "WhatsApp number. Call once at the start of a conversation."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "select_child",
        "description": "Set which registered child this session is about.",
        "parameters": {"type": "object", "properties": {
            "student_id": {"type": "string",
                           "description": "student_id from find_children"}},
            "required": ["student_id"]},
    }},
    {"type": "function", "function": {
        "name": "register_child",
        "description": ("Register a new child. Only call once you have the name, "
                        "level and gender, and the parent has confirmed them."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "First name, capitalised"},
            "level": {"type": "string", "enum": LEVELS},
            "gender": {"type": "string", "enum": ["Male", "Female"]}},
            "required": ["name", "level", "gender"]},
    }},
    {"type": "function", "function": {
        "name": "list_topics",
        "description": ("Maths topics available for the selected child. Call "
                        "before naming or confirming any topic."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "list_subtopics",
        "description": "Subtopics under one topic. Only when the parent asks to narrow down.",
        "parameters": {"type": "object", "properties": {
            "topic_id": {"type": "string"}}, "required": ["topic_id"]},
    }},
    {"type": "function", "function": {
        "name": "create_practice",
        "description": ("Generate the practice set and return its link. Call only "
                        "after confirming with the parent. Question count and type "
                        "are set by the system."),
        "parameters": {"type": "object", "properties": {
            "topic_ids": {"type": "array", "items": {"type": "string"},
                          "description": ("ALL requested topic_ids from list_topics, in "
                                          "one list. e.g. topics 2, 5 and 6 becomes "
                                          "[\"2\",\"5\",\"6\"], not three calls.")},
            "subtopic_id": {"type": "string",
                            "description": "Optional, only if one was chosen"},
            "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}},
            "required": ["topic_ids", "difficulty"]},
    }},
    {"type": "function", "function": {
        "name": "create_practice_pdf",
        "description": ("Generate a printable PDF paper with an answer key, "
                        "delivered as a WhatsApp document. PDF papers "
                        "are NOT marked and produce NO report. Only call after the "
                        "parent has confirmed they want the PDF and you have told "
                        "them it will not be marked. The system sends the file "
                        "itself; do not describe a link."),
        "parameters": {"type": "object", "properties": {
            "topic_ids": {"type": "array", "items": {"type": "string"},
                          "description": ("ALL requested topic_ids from list_topics, "
                                          "in one list, same rule as create_practice.")},
            "subtopic_id": {"type": "string",
                            "description": "Optional, only if one was chosen"},
            "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}},
            "required": ["topic_ids", "difficulty"]},
    }},
    {"type": "function", "function": {
        "name": "get_report",
        "description": ("Fetch the child's completed practice reports. Call when "
                        "the parent asks how their child did, for results, scores, "
                        "or the report. Returns the latest report link, which the "
                        "system appends to your message automatically."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "end_conversation",
        "description": "The parent is done. Clears the session after your goodbye.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


# ---------------------------------------------------------------- tool impls
async def t_find_children(s: dict, phone: str) -> dict:
    local = phone[2:] if phone.startswith("65") else phone
    s["phone_local"] = local

    found = await asyncio.to_thread(registry.check_student_exists, local)
    err = registry.last_error

    if err:
        cached = check_existing_student_local(local)
        if cached:
            log.warning("Platform lookup failed (%s); using local cache", err)
            found = cached
        else:
            log.error("Platform lookup failed (%s) and no local cache", err)
            return {"error": "lookup_unavailable", "reason": err,
                    "message": ("Student records are unreachable right now. "
                                "Do NOT register anyone. Apologise briefly and "
                                "ask them to try again in a few minutes.")}

    if not found:
        return {"children": [],
                "note": "No children registered to this number yet."}

    if len(found) == 1:
        s["child"] = found[0]
        s["topics"] = []

    return {"children": [{"student_id": str(c.get("student_id", "")),
                          "name": c.get("name", ""),
                          "level": c.get("primary_level", "")} for c in found],
            "auto_selected": len(found) == 1}


async def t_select_child(s: dict, student_id: str) -> dict:
    local = s.get("phone_local", "")
    found = await asyncio.to_thread(registry.check_student_exists, local)
    if registry.last_error:
        found = check_existing_student_local(local)
    found = found or []
    for c in found:
        if str(c.get("student_id")) == str(student_id):
            s["child"] = c
            s["topics"] = []
            return {"ok": True, "name": c.get("name"),
                    "level": c.get("primary_level")}
    return {"error": "not_found",
            "message": "That child is not registered to this number."}


async def t_register_child(s: dict, name: str, level: str, gender: str) -> dict:
    level = normalize_level(level)
    if not level:
        return {"error": "bad_level", "valid": LEVELS,
                "message": "Ask which primary level, 1 to 6."}

    clean_name = name.strip()
    phone = s.get("phone_local", "")
    new = await asyncio.to_thread(registry.register_student,
                                  clean_name, level, gender, phone)
    err = registry.last_error

    if not new and err == "exists":
        log.info("409 for %s, re-querying to adopt the existing record", clean_name)
        existing = await asyncio.to_thread(registry.check_student_exists, phone)
        if not registry.last_error and existing:
            match = next((c for c in existing
                          if c.get("name", "").strip().lower() == clean_name.lower()),
                         None)
            if match is None and len(existing) == 1:
                match = existing[0]
            if match:
                s["child"] = match
                s["topics"] = []
                log.info("Adopted existing student %s (%s)",
                         match.get("name"), match.get("student_id"))
                return {"ok": True, "already_existed": True,
                        "student_id": str(match.get("student_id")),
                        "name": match.get("name"),
                        "level": match.get("primary_level"),
                        "message": ("Already registered. Greet them by name and "
                                    "carry on, do not mention the duplicate.")}
            return {"error": "ambiguous_existing",
                    "children": [{"student_id": str(c.get("student_id")),
                                  "name": c.get("name"),
                                  "level": c.get("primary_level")}
                                 for c in existing],
                    "message": "Already registered. Ask which child this is."}
        return {"error": "exists_but_unreadable",
                "message": ("That child already exists but the records are "
                            "unreachable. Apologise and ask them to try shortly.")}

    if not new:
        log.error("Registration failed for %s (%s)", clean_name, err)
        save_student_local(clean_name, level, gender, phone,
                           f"S{datetime.now().strftime('%Y%m%d%H%M%S')}")
        return {"error": "registration_unavailable", "reason": err,
                "message": ("Could not reach the student records. Their details "
                            "are saved. Apologise and ask them to try again in a "
                            "few minutes. Do not offer practice yet.")}

    student_id = new.get("student_id", "")
    save_student_local(clean_name, level, gender, phone, student_id)

    returned_name = new.get("name", clean_name)
    s["child"] = {"student_id": student_id, "name": returned_name,
                  "primary_level": level, "gender": gender, "phone": phone}
    s["topics"] = []
    log.info("Registered new student %s (%s)", returned_name, student_id)
    return {"ok": True, "student_id": str(student_id),
            "name": returned_name, "level": level}


async def t_list_topics(s: dict) -> dict:
    if not s.get("child"):
        return {"error": "no_child", "message": "Select or register a child first."}
    if not usable_student_id(s["child"].get("student_id")):
        return {"error": "child_not_on_platform",
                "message": ("This child was saved locally during an outage and "
                            "is not on the platform yet. Apologise and ask them "
                            "to try again shortly.")}
    if not s.get("topics"):
        s["topics"] = await asyncio.to_thread(
            _get_topics, s["child"].get("student_id")) or []
    if not s["topics"]:
        return {"error": "unavailable",
                "message": "Topics could not be loaded right now."}
    return {"topics": [{"topic_id": str(t.get("topic_id")),
                        "topic_name": t.get("topic_name")} for t in s["topics"]]}


async def t_list_subtopics(s: dict, topic_id: str) -> dict:
    if not s.get("child"):
        return {"error": "no_child"}
    subs = await asyncio.to_thread(
        _get_subtopics, s["child"].get("student_id"), topic_id)
    if not subs:
        return {"subtopics": [],
                "note": "None available, use the broad topic instead."}
    return {"subtopics": [{"subtopic_id": str(x.get("subtopic_id", x.get("id", ""))),
                           "subtopic_name": x.get("subtopic_name", x.get("name", ""))}
                          for x in subs]}


def _validate_practice_request(s: dict, topic_ids: list, difficulty: str):
    if not s.get("child"):
        return {"error": "no_child", "message": "Select or register a child first."}
    if not usable_student_id(s["child"].get("student_id")):
        return {"error": "child_not_on_platform",
                "message": ("Not on the platform yet, so practice cannot be "
                            "generated. Apologise and ask them to try shortly.")}
    if difficulty not in DIFFICULTY_EMOJI:
        return {"error": "bad_difficulty", "valid": list(DIFFICULTY_EMOJI)}
    return None


async def _validate_topics(s: dict, topic_ids: list):
    if not s.get("topics"):
        s["topics"] = await asyncio.to_thread(
            _get_topics, s["child"].get("student_id")) or []
    valid = {str(t.get("topic_id")) for t in s["topics"]}
    unknown = [t for t in map(str, topic_ids) if t not in valid]
    if unknown:
        return {"error": "unknown_topic", "unknown": unknown,
                "valid_topics": [{"topic_id": str(t.get("topic_id")),
                                  "topic_name": t.get("topic_name")}
                                 for t in s["topics"]]}
    return None


async def t_create_practice(s: dict, topic_ids: list, difficulty: str,
                            subtopic_id: str | None = None) -> dict:
    err = _validate_practice_request(s, topic_ids, difficulty)
    if err:
        return err
    err = await _validate_topics(s, topic_ids)
    if err:
        return err

    child = s["child"]
    level = normalize_level(child.get("primary_level", ""))
    if not level:
        log.warning("No usable level for student %s (raw=%r), using generic paper",
                    child.get("student_id"), child.get("raw_level"))
    picked = defaults_for(level, difficulty)
    ids = [subtopic_id] if subtopic_id else list(map(str, topic_ids))

    url = await asyncio.to_thread(
        _generate_worksheet_url, child.get("student_id"), ids, difficulty,
        picked["count"], picked["type"])

    if not url:
        return {"error": "generation_failed",
                "message": "Could not generate just now, ask them to try again."}

    phone = s.get("phone_full", "")
    if phone:
        start_report_watch(child.get("student_id"), phone,
                           child.get("name") or "Your child")

    return {"link": url, "difficulty": difficulty,
            "emoji": DIFFICULTY_EMOJI.get(difficulty, ""),
            "child": child.get("name")}


async def t_create_practice_pdf(s: dict, topic_ids: list, difficulty: str,
                                subtopic_id: str | None = None) -> dict:
    err = _validate_practice_request(s, topic_ids, difficulty)
    if err:
        return err

    if not SUPPORTS_PDF:
        return {"error": "pdf_unavailable",
                "message": ("PDF papers are switched off right now. Apologise in "
                            "one short line and offer the normal practice link "
                            "instead.")}

    err = await _validate_topics(s, topic_ids)
    if err:
        return err

    child = s["child"]
    phone = s.get("phone_full", "")
    if not phone:
        return {"error": "internal_error",
                "message": "Could not deliver. Ask them to try again."}

    level = normalize_level(child.get("primary_level", "")) or "Primary"
    picked = defaults_for(level, difficulty)
    ids = [subtopic_id] if subtopic_id else list(map(str, topic_ids))

    await send_message(phone, "Putting the paper together now, one moment 📄")

    url = await asyncio.to_thread(
        _generate_worksheet_url, child.get("student_id"), ids, difficulty,
        picked["count"], picked["type"], 1)
    if not url:
        return {"error": "generation_failed",
                "message": ("Could not generate the PDF just now. Offer the "
                            "normal practice link instead.")}

    name = (child.get("name") or "Practice").strip()
    safe_name = re.sub(r"[^A-Za-z0-9 _-]", "", name) or "Practice"
    filename = f"{safe_name} - {level} {difficulty} Practice.pdf"
    caption = (f"{name}'s practice paper. PDF papers aren't marked and "
               "don't count towards reports.")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    delivered = False
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
            r = await http.get(url)
        if r.status_code == 200 and (r.content[:5] == b"%PDF-" or
                                     "pdf" in r.headers.get("content-type", "").lower()):
            with open(tmp.name, "wb") as f:
                f.write(r.content)
            delivered = await send_document(phone, tmp.name, filename, caption)
        else:
            log.info("PDF url did not serve a raw pdf (status %s, type %s); "
                     "falling back to link", r.status_code,
                     r.headers.get("content-type"))
    except Exception:
        log.exception("PDF fetch/send failed; falling back to link")
    finally:
        with suppress(OSError):
            os.unlink(tmp.name)

    if delivered:
        return {"ok": True, "delivered": "pdf_document", "child": name,
                "message": ("PDF sent as a document in the chat. Confirm in one "
                            "short line and remind them PDF papers are not "
                            "marked and no report will come.")}

    return {"link": url, "delivered": "pdf_link", "child": name,
            "difficulty": difficulty,
            "message": ("Could not attach the file directly, so the PDF "
                        "download link will be added to your message. Say the "
                        "paper is ready to download, and remind them it is "
                        "not marked.")}


async def t_get_report(s: dict) -> dict:
    if not s.get("child"):
        return {"error": "no_child", "message": "Select or register a child first."}
    if not usable_student_id(s["child"].get("student_id")):
        return {"error": "child_not_on_platform",
                "message": ("Not on the platform yet. Apologise and ask them "
                            "to try again shortly.")}
    reports = await asyncio.to_thread(
        _get_student_reports, s["child"].get("student_id"))
    if reports is None:
        return {"error": "unavailable",
                "message": "Reports could not be loaded right now. Apologise "
                           "and ask them to try again in a few minutes."}
    if not reports:
        return {"reports": 0,
                "note": ("No completed practice yet. Tell them the report "
                         "appears here once their child finishes a practice "
                         "set through the link.")}
    latest = reports[0]
    return {"link": latest.get("report_link"),
            "title": latest.get("homework_quiz_title"),
            "total_reports": len(reports),
            "child": s["child"].get("name"),
            "message": ("Latest report found. Write one short line like "
                        "\'Here\'s how NAME did on their latest practice\' - "
                        "the link is appended automatically, never write it "
                        "yourself.")}


async def t_end_conversation(s: dict) -> dict:
    s["_end"] = True
    return {"ok": True}


DISPATCH = {
    "find_children": t_find_children,
    "select_child": t_select_child,
    "register_child": t_register_child,
    "list_topics": t_list_topics,
    "list_subtopics": t_list_subtopics,
    "create_practice": t_create_practice,
    "create_practice_pdf": t_create_practice_pdf,
    "get_report": t_get_report,
    "end_conversation": t_end_conversation,
}


async def run_tool(name: str, args: dict, s: dict, phone: str) -> dict:
    fn = DISPATCH.get(name)
    if not fn:
        return {"error": "unknown_tool", "name": name}
    try:
        if name == "find_children":
            return await fn(s, phone)
        return await fn(s, **args)
    except TypeError as e:
        return {"error": "bad_arguments", "detail": str(e)[:200]}
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": "internal_error", "detail": str(e)[:200]}


# ---------------------------------------------------------------- agent loop
async def agent_turn(phone: str, text: str) -> None:
    s = session_for(phone)
    s["phone_full"] = phone

    history = sanitize_history(s["messages"])
    history.append({"role": "user", "content": text})
    log.info("IN  %s %r", phone[-4:], text[:80])

    pending_links: list[str] = []

    for hop in range(MAX_TOOL_HOPS):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=TOOLS,
                temperature=0.6,
            )
        except Exception as e:
            log.error("LLM error: %s | messages=%s",
                      e, json.dumps(history, default=str)[:2000])
            s["messages"] = trim_history(history)
            await send_message(phone, "Something went wrong on our end. "
                                      "Please try again in a moment.")
            return

        msg = resp.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            body = clean_reply(msg.content or "")
            if pending_links:
                body = (body + "\n\n" + "\n".join(pending_links)).strip()
            if body:
                await send_message(phone, body)
            pending_links.clear()
            break

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await run_tool(call.function.name, args, s, phone)
            log.info("TOOL %s -> %s", call.function.name, list(result)[:4])
            history.append({"role": "tool", "tool_call_id": call.id,
                            "content": json.dumps(result)})

            link = result.get("link")
            if call.function.name in ("create_practice", "create_practice_pdf",
                                      "get_report") and link:
                if link not in pending_links:
                    pending_links.append(link)
    else:
        body = "Sorry, I got a bit stuck there. Could you tell me that again?"
        if pending_links:
            body = ("Here's the practice.\n\n" + "\n".join(pending_links))
        pending_links.clear()
        await send_message(phone, body)

    s["messages"] = trim_history(history)

    if s.get("_end"):
        SESSIONS.pop(phone, None)

    log.info("OUT %s", phone[-4:])


# ---------------------------------------------------------------- webhook
def verify_signature(body: bytes, header: str | None) -> bool:
    if not APP_SECRET:
        log.warning("META_APP_SECRET not set — signature check skipped")
        return True
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


@app.get("/")
async def verify(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=p.get("hub.challenge"), media_type="text/plain")
    return Response(status_code=403)


@app.get("/health")
async def health():
    return {"ok": True, "sessions": len(SESSIONS)}


@app.post("/")
async def receive(request: Request, background: BackgroundTasks):
    raw = await request.body()
    if not verify_signature(raw, request.headers.get("X-Hub-Signature-256")):
        log.warning("Rejected unsigned webhook POST")
        return Response(status_code=403)

    body = json.loads(raw)
    try:
        value = body["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError):
        return {"status": "ignored"}

    for st in value.get("statuses", []):
        log.info("STATUS %s -> %s (%s) %s", st.get("status"),
                 st.get("recipient_id"), st.get("id"), st.get("errors", ""))

    for msg in value.get("messages", []):
        if already_processed(msg.get("id")):
            continue

        if msg.get("type") == "text":
            text = (msg.get("text") or {}).get("body", "").strip()
        elif msg.get("type") == "interactive":
            i = msg["interactive"]
            text = (i.get("button_reply") or i.get("list_reply") or {}).get("title", "")
        else:
            background.add_task(
                send_message, msg["from"],
                "I can only read text at the moment. Tell me your child's level "
                "and what they'd like to practise.")
            continue

        if text:
            background.add_task(agent_turn, msg["from"], text)

    return {"status": "ok"}


@app.post("/assessment-completed")
async def assessment_completed(request: Request, background: BackgroundTasks):
    if not PLATFORM_CALLBACK_SECRET or \
            request.headers.get("X-Callback-Secret") != PLATFORM_CALLBACK_SECRET:
        log.warning("Rejected report callback (bad or missing secret)")
        return Response(status_code=403)

    try:
        data = await request.json()
    except Exception:
        return Response(status_code=400)

    phone = re.sub(r"\D", "", str(data.get("phone", "")))
    if phone and not phone.startswith("65"):
        phone = "65" + phone
    student_name = (data.get("student_name") or "").strip() or "Your child"
    student_name = re.sub(r"\bLast\b", "", student_name, flags=re.IGNORECASE).strip() or "Your child"
    report_link = (data.get("report_link") or "").strip()

    if not phone or not report_link:
        log.warning("Report callback missing phone or report_link: %s",
                    {k: data.get(k) for k in ("student_id", "phone",
                                              "report_link")})
        return {"status": "ignored", "reason": "missing phone or report_link"}

    background.add_task(send_report_template, phone, student_name, report_link)
    log.info("Report queued for %s (student %s)", phone[-4:],
             data.get("student_id"))
    return {"status": "ok"}
