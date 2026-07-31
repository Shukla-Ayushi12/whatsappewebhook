import os
import json
import hmac
import hashlib
import asyncio
import logging
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
APP_SECRET = os.getenv("META_APP_SECRET")          # NEW — see verify_signature
GRAPH_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

registry = StudentRegistry(
    api_key=os.getenv("WHATSPREP_API_KEY"),
    base_url="https://latam.whatsprep.com/api",
)

DB_FILE = "students.json"

# Set True once /generate-assessment is confirmed to accept question_count
# and question_type. While False the bot still picks values internally but
# does not promise them to the parent.
SUPPORTS_QUESTION_OPTIONS = False

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

# phone -> {"messages": [...], "child": {...} | None, "topics": [...] }
SESSIONS: dict[str, dict] = {}
PROCESSED_IDS: set[str] = set()


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
    students = load_database()
    students.append({
        "student_id": student_id, "name": name, "primary_level": level,
        "gender": gender, "phone": phone,
        "joined_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(DB_FILE, "w") as f:
        json.dump(students, f, indent=4)


def check_existing_student_local(phone: str) -> list | None:
    """Local cache. Only used when the platform is unreachable.

    Rows written by the degraded path carry an "S..." id, which the platform
    endpoints cannot use, so they are normalised here and filtered at the
    point of use rather than silently failing later.
    """
    matches = []
    for s in load_database():
        if s.get("phone", "").lstrip("+65") != phone:
            continue
        row = dict(s)
        row["primary_level"] = normalize_level(row.get("primary_level"))
        matches.append(row)
    return matches or None


def usable_student_id(student_id) -> bool:
    """Platform ids are numeric. Local fallback ids look like S20260731...."""
    try:
        int(student_id)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- platform api
# Unchanged from your version. Sync `requests` calls are wrapped in
# asyncio.to_thread at the call site so they don't block the event loop.
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
                            count=None, qtype=None) -> str | None:
    try:
        sid = int(student_id)
        tids = [int(t) for t in topic_ids]
    except (ValueError, TypeError):
        return None
    payload = {"topic_ids": tids, "student_id": sid, "difficulty": difficulty}
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
            url = d.get("assessment_url") or d.get("assessment_ur") or d.get("url")
            log.info("Generated worksheet for student %s topics %s: %s",
                     sid, tids, bool(url))
            return url
        log.warning("Generate failed: %s %s", r.status_code, r.text)
    except Exception as e:
        log.error("Generate error: %s", e)
    return None


# ---------------------------------------------------------------- session
def session_for(phone: str) -> dict:
    if phone not in SESSIONS:
        SESSIONS[phone] = {"messages": [], "child": None, "topics": []}
    return SESSIONS[phone]


# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = """\
You are the WhatsPrep assistant. You help Singapore parents start Maths \
practice for their primary school child, over WhatsApp. Maths only. Never \
offer or discuss other subjects.

HOW THE PRODUCT WORKS
You do not send questions in the chat. You generate a practice set and send \
back a link the child opens. Never write maths questions yourself.

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
no markdown, no bullet characters, no headings. Many parents are not \
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
                          "description": "One or more topic_ids from list_topics"},
            "subtopic_id": {"type": "string",
                            "description": "Optional, only if one was chosen"},
            "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]}},
            "required": ["topic_ids", "difficulty"]},
    }},
    {"type": "function", "function": {
        "name": "end_conversation",
        "description": "The parent is done. Clears the session after your goodbye.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


# ---------------------------------------------------------------- tool impls
# Every tool reads the phone and active child from the session, never from
# model arguments. The model cannot address another parent's children.

async def t_find_children(s: dict, phone: str) -> dict:
    local = phone[2:] if phone.startswith("65") else phone
    s["phone_local"] = local

    found = await asyncio.to_thread(registry.check_student_exists, local)
    err = registry.last_error

    # The lookup FAILED. Never treat this as "new parent" or we register a
    # duplicate child every time the platform blips or the JWT expires.
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

    # The lookup SUCCEEDED and returned nothing. Any local row here is a
    # leftover from a failed registration and has an unusable id, so ignore it.
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

    phone = s.get("phone_local", "")
    new = await asyncio.to_thread(registry.register_student,
                                  name, level, gender, phone)
    err = registry.last_error

    # 409: they already exist on the platform. Adopt the real record rather
    # than minting a local id that can never generate a worksheet.
    if not new and err == "exists":
        log.info("409 for %s, re-querying to adopt the existing record", name)
        existing = await asyncio.to_thread(registry.check_student_exists, phone)
        if not registry.last_error and existing:
            match = next((c for c in existing
                          if c.get("name", "").strip().lower() == name.strip().lower()),
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

    # Any other failure. Keep the details locally so nothing is lost, but do
    # NOT set the active child: a local id cannot generate a worksheet.
    if not new:
        log.error("Registration failed for %s (%s)", name, err)
        save_student_local(name, level, gender, phone,
                           f"S{datetime.now().strftime('%Y%m%d%H%M%S')}")
        return {"error": "registration_unavailable", "reason": err,
                "message": ("Could not reach the student records. Their details "
                            "are saved. Apologise and ask them to try again in a "
                            "few minutes. Do not offer practice yet.")}

    student_id = new.get("student_id", "")
    save_student_local(name, level, gender, phone, student_id)
    s["child"] = {"student_id": student_id, "name": name,
                  "primary_level": level, "gender": gender, "phone": phone}
    s["topics"] = []
    log.info("Registered new student %s (%s)", name, student_id)
    return {"ok": True, "student_id": str(student_id),
            "name": name, "level": level}


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


async def t_create_practice(s: dict, topic_ids: list, difficulty: str,
                            subtopic_id: str | None = None) -> dict:
    if not s.get("child"):
        return {"error": "no_child", "message": "Select or register a child first."}
    if not usable_student_id(s["child"].get("student_id")):
        return {"error": "child_not_on_platform",
                "message": ("Not on the platform yet, so practice cannot be "
                            "generated. Apologise and ask them to try shortly.")}
    if difficulty not in DIFFICULTY_EMOJI:
        return {"error": "bad_difficulty", "valid": list(DIFFICULTY_EMOJI)}

    # Validate topic ids against the real list. Models occasionally invent one.
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
    return {"link": url, "difficulty": difficulty,
            "emoji": DIFFICULTY_EMOJI.get(difficulty, ""),
            "child": child.get("name")}


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
    history = s["messages"]
    history.append({"role": "user", "content": text})
    log.info("IN  %s %r", phone[-4:], text[:80])

    sent_link = False

    for hop in range(MAX_TOOL_HOPS):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=TOOLS,
                temperature=0.6,
            )
        except Exception as e:
            log.error("LLM error: %s", e)
            await send_message(phone, "Something went wrong on our end. "
                                      "Please try again in a moment.")
            return

        msg = resp.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            if msg.content:
                await send_message(phone, msg.content)
            break

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await run_tool(call.function.name, args, s, phone)
            log.info("TOOL %s -> %s", call.function.name,
                     list(result)[:4])
            history.append({"role": "tool", "tool_call_id": call.id,
                            "content": json.dumps(result)})

            # Send the link on its own so WhatsApp renders a preview
            if call.function.name == "create_practice" and result.get("link"):
                await send_message(phone, result["link"])
                sent_link = True
    else:
        await send_message(phone, "Sorry, I got a bit stuck there. "
                                  "Could you tell me that again?")

    s["messages"] = history[-MAX_HISTORY:]

    if s.get("_end"):
        SESSIONS.pop(phone, None)

    log.info("OUT %s link=%s", phone[-4:], sent_link)


# ---------------------------------------------------------------- webhook
def verify_signature(body: bytes, header: str | None) -> bool:
    """Your previous version accepted any POST to /. This closes that."""
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

    # Your version handled msgs[0] only; this drains the batch.
    for msg in value.get("messages", []):
        msg_id = msg.get("id")
        if msg_id in PROCESSED_IDS:
            continue
        PROCESSED_IDS.add(msg_id)
        if len(PROCESSED_IDS) > 1000:
            PROCESSED_IDS.clear()

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

