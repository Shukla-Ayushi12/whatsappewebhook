import re
import json
import os
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from student_registry import StudentRegistry
from dotenv import load_dotenv
import requests

# --- Load environment variables ---
load_dotenv()

# --- OpenAI setup ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Google Sheets setup ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_URL = os.getenv("SHEET_URL")
CREDENTIALS_FILE = "credentials.json"

PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY")
if not PLATFORM_API_KEY:
    raise RuntimeError("PLATFORM_API_KEY not set in .env")

registry = StudentRegistry(
    api_key=PLATFORM_API_KEY,
    base_url=os.getenv("WHATSPREP_BASE_URL", "https://latam.whatsprep.com/api")
)
# --- Local database file ---
DB_FILE = "students.json"

# --- Session memory ---
session = {}

# --- Google Sheets client (optional) ---
students_sheet = None


# ─────────────────────────────────────────
# GOOGLE SHEETS FUNCTIONS
# ─────────────────────────────────────────

def get_sheets():
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_url(SHEET_URL)
        sheet = spreadsheet.worksheet("Students")
        if not sheet.row_values(1):
            sheet.append_row(["Student ID", "Name", "Primary Level", "Gender", "Phone", "Date Joined"])
        print("📊 Connected to Google Sheets!")
        return sheet
    except FileNotFoundError:
        print("⚠️  credentials.json not found. Skipping Google Sheets.")
        return None
    except gspread.exceptions.SpreadsheetNotFound:
        print("⚠️  Google Sheet not found. Skipping Google Sheets.")
        return None
    except Exception as e:
        print(f"⚠️  Could not connect to Google Sheets: {e}. Continuing without it.")
        return None


def save_student_to_sheets(sheet, name, level, gender, phone):
    if not sheet:
        return None
    try:
        all_students = sheet.get_all_records()
        student_id = f"S{len(all_students) + 1:04d}"
        sheet.append_row([student_id, name, level, gender, phone,
                          datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        print(f"📊 Student saved to Google Sheets (ID: {student_id})")
        return student_id
    except Exception as e:
        print(f"⚠️  Error saving to Google Sheets: {e}")
        return None


# ─────────────────────────────────────────
# LOCAL DATABASE FUNCTIONS
# ─────────────────────────────────────────

def load_database() -> list:
    try:
        if not os.path.exists(DB_FILE):
            return []
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️  Local database corrupted. Starting fresh.")
        return []
    except Exception as e:
        print(f"⚠️  Error loading local database: {e}")
        return []


def save_database(data: list):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"⚠️  Error saving to local database: {e}")


def check_existing_student_local(phone: str) -> list | None:
    try:
        students = load_database()
        formatted_phone = f"+65{phone}" if not phone.startswith("+") else phone
        matches = []
        for student in students:
            stored_phone = student.get("phone", "")
            stored_formatted = f"+65{stored_phone}" if not stored_phone.startswith("+") else stored_phone
            if stored_formatted == formatted_phone or stored_phone == phone:
                matches.append(student)
        return matches if matches else None
    except Exception as e:
        print(f"⚠️  Error checking local database: {e}")
        return None


def find_student_locally(identifier_type: str, value: str) -> dict | None:
    try:
        students = load_database()
        value_lower = value.lower()
        for student in students:
            if identifier_type == "NAME":
                if student.get("name", "").lower() == value_lower:
                    return student
            elif identifier_type == "ID":
                if str(student.get("student_id", "")) == str(value):
                    return student
        return None
    except Exception as e:
        print(f"⚠️  Error searching local database: {e}")
        return None


def save_student_local(name, level, gender, phone, student_id):
    try:
        students = load_database()
        students.append({
            "student_id": student_id,
            "name": name,
            "primary_level": level,
            "gender": gender,
            "phone": phone,
            "joined_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_database(students)
    except Exception as e:
        print(f"⚠️  Error saving student locally: {e}")


# ─────────────────────────────────────────
# TOPICS & SUBTOPICS API
# ─────────────────────────────────────────

def get_topics(level: str, subject: str) -> list | None:
    print(f"📚 Fetching topics for {subject} {level}...")

    subject_map = {
        "Math": "Mathematics",
        "Science": "Science",
        "English": "English"
    }
    api_subject = subject_map.get(subject, subject)

    raw_id = session.get("student_id", "")
    try:
        student_id = int(raw_id)
    except (ValueError, TypeError):
        print(f"⚠️  Student not on platform (local ID: {raw_id}). Cannot fetch topics.")
        print("💡 Please register this student on the platform first.")
        return None

    try:
        payload = {
            "student_id": student_id,
            "subject": api_subject
        }
        print(f"📦 Sending payload: {payload}")

        response = requests.post(
            "https://latam.whatsprep.com/api/topics",
            headers=registry.headers,
            json=payload,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📡 Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            topics = data.get("data", [])
            if not topics:
                print("⚠️  API returned empty topics list.")
            return topics if topics else None
        else:
            print(f"⚠️  Could not fetch topics: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Error fetching topics: {e}")
        return None


def get_subtopics(topic_id: str) -> list | None:
    print(f"📚 Fetching subtopics for topic {topic_id}...")

    raw_id = session.get("student_id", "")
    try:
        student_id = int(raw_id)
    except (ValueError, TypeError):
        print(f"⚠️  Student not on platform. Cannot fetch subtopics.")
        return None

    try:
        payload = {
            "student_id": student_id,
            "topic_id": int(topic_id)
        }
        print(f"📦 Sending payload: {payload}")

        response = requests.post(
            "https://latam.whatsprep.com/api/topics",
            headers=registry.headers,
            json=payload,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📡 Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            subtopics = data.get("data", [])
            if not subtopics:
                print("⚠️  API returned empty subtopics list.")
            return subtopics if subtopics else None
        else:
            print(f"⚠️  Could not fetch subtopics: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Error fetching subtopics: {e}")
        return None


def generate_worksheet_url(topic_id: str, level: str, subject: str,
                           difficulty: str, student_id: str = "") -> str | None:
    """Generate an assessment using /generate-assessment.
    Note: level and subject are no longer needed — derived from topic_id and student_id."""
    print(f"⏳ Generating worksheet...")

    raw_id = session.get("student_id", "")
    try:
        student_id_int = int(raw_id)
    except (ValueError, TypeError):
        print(f"⚠️  Student not on platform (local ID: {raw_id}). Cannot generate assessment.")
        return None

    try:
        topic_id_int = int(topic_id)
    except (ValueError, TypeError):
        print(f"⚠️  Invalid topic_id: {topic_id}. Cannot generate assessment with multiple topics.")
        return None

    try:
        payload = {
            "topic_id": topic_id_int,
            "student_id": student_id_int,
            "difficulty": difficulty
        }
        print(f"📦 Sending payload: {payload}")

        response = requests.post(
            "https://latam.whatsprep.com/api/generate-assessment",
            headers=registry.headers,
            json=payload,
            timeout=10
        )
        print(f"📡 Status: {response.status_code}")
        print(f"📡 Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            response_data = data.get("data", {})
            print(f"🔑 Available keys in response_data: {list(response_data.keys())}")

            url = (
                response_data.get("assessment_url")
                or response_data.get("assessment_ur")
                or response_data.get("url")
            )

            if not url:
                print("⚠️  No URL returned in response.")
            return url
        else:
            print(f"⚠️  Could not generate assessment: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Error generating assessment: {e}")
        return None
# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def call_llm(messages: list, max_tokens: int = 50) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  Error communicating with AI: {e}")
        return ""


def safe_input(prompt: str) -> str:
    user_input = input(prompt).strip()

    wants_to_exit = call_llm([{
        "role": "user",
        "content": f"""Does this message CLEARLY mean the person wants to leave, quit, or stop?

Message: "{user_input}"

Rules:
- Only return "yes" if the message is unambiguously about leaving/quitting/stopping.
- If the message looks like a typo, partial word, or could be an attempt at answering the question, return "no".
- If you are unsure or it's ambiguous, return "no" — when in doubt, do NOT treat it as an exit.
- Single short words that aren't clearly exit-related (e.g. "mat", "yes", "ok", "1", "Primary 4") are "no".

Examples:
"exit" -> yes
"quit" -> yes
"bye" -> yes
"leave" -> yes
"goodbye" -> yes
"i want to stop" -> yes
"no thanks" -> yes
"i'm done" -> yes
"mat" -> no
"math" -> no
"hi" -> no
"yes" -> no
"ok" -> no
"1" -> no
"Primary 4" -> no

Answer with only "yes" or "no"."""
    }], max_tokens=5)

    if wants_to_exit and wants_to_exit.lower() == "yes":
        name = session.get("name", "")
        if name:
            print(f"\n👋 Thank you for using Whatsprep, {name}! Goodbye!")
        else:
            print("\n👋 Goodbye! Come back anytime.")
        exit()

    return user_input


# --- Yes/No parsing: cheap local check first, LLM only when unclear ---

YES = {"yes","y","yup","yeah","yea","ya","yep","yah","sure","ok","okay","okok",
       "correct","right","true","confirm","confirmed","that's right","thats right",
       "yes please","correct la","can","👍","✅"}

NO  = {"no","n","nope","nah","naw","not","wrong","incorrect","false",
       "not really","no lah","cannot","❌"}


def parse_yes_no(text: str):
    """Fast local check. Returns True / False / None (unclear)."""
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
    result = call_llm([
        {"role": "system", "content":
         "You classify a parent's reply to a yes/no question. "
         "Reply with exactly one word: YES, NO, or UNCLEAR. "
         "Treat affirmations ('that's right', 'she is', 'correct') as YES "
         "and denials ('not quite', 'wrong one') as NO."},
        {"role": "user", "content": text},
    ], max_tokens=5)
    a = (result or "").strip().upper()
    return True if a == "YES" else False if a == "NO" else None


def ask_yes_no(prompt: str) -> bool:
    """Ask a yes/no question until we get a clear answer."""
    while True:
        raw = safe_input(prompt)
        ans = parse_yes_no(raw)
        if ans is None:
            ans = ai_yes_no(raw)
        if ans is not None:
            return ans
        print("Sorry, I didn't quite catch that — is that a yes or a no?")
    if wants_to_exit and wants_to_exit.lower() == "yes":
        name = session.get("name", "")
        if name:
            print(f"\n👋 Thank you for using Whatsprep, {name}! Goodbye!")
        else:
            print("\n👋 Goodbye! Come back anytime.")
        exit()

    return user_input

def ask_with_retries(prompt: str, validate_fn, error_msg: str, max_tries=3) -> str:
    for attempt in range(max_tries):
        user_input = safe_input(prompt)
        result = validate_fn(user_input)
        if result != "Unknown":
            return result
        remaining = max_tries - attempt - 1
        if remaining > 0:
            print(f"⚠️  {error_msg} ({remaining} tries left)")
        else:
            print("\n😔 Having trouble? Please contact us at support@whatsprep.com")
            exit()


def pick_student(students: list) -> dict | None:
    if len(students) == 1:
        return students[0]
    print(f"\n👨‍👩‍👧‍👦 We found {len(students)} children under this number:")
    for i, s in enumerate(students, 1):
        level = s.get("primary_level") or "Unknown level"
        print(f"  {i}. {s['name']} — {level}")
    while True:
        try:
            choice = safe_input("\nWhich child is this for? Enter the number: ")
            index = int(choice) - 1
            if 0 <= index < len(students):
                return students[index]
            print(f"⚠️  Please enter a number between 1 and {len(students)}.")
        except ValueError:
            print("⚠️  Please enter a number.")


def extract_student_identifier(text: str) -> tuple:
    result = call_llm([{
        "role": "user",
        "content": f"""Extract either a student name or student ID from this message: "{text}"

Rules:
- If you find a student ID (starts with S followed by numbers e.g. S0012, or a plain number like 15), return: ID|<the_id>
- If you find a child's name, return: NAME|<the_name>
- If neither found, return: UNKNOWN|UNKNOWN

Examples:
"hi i want math MCQ for my child Ayushi" → NAME|Ayushi
"i want math for S0012" → ID|S0012
"give me math questions for student id 15" → ID|15
"hi" → UNKNOWN|UNKNOWN"""
    }], max_tokens=20)
    try:
        kind, value = result.strip().split("|", 1)
        return kind.strip().upper(), value.strip()
    except Exception:
        return "UNKNOWN", "UNKNOWN"


# ─────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────

def validate_name(text: str) -> str:
    result = call_llm([{
        "role": "user",
        "content": f"""Extract only the child's name from this input: "{text}"
Rules:
- Return ONLY the name, properly capitalised
- If no valid name found, return "Unknown"
- No extra words or punctuation

Examples:
"his name is john" → John
"she is ayushi" → Ayushi
"abc123" → Unknown"""
    }])
    if not result or result == "Unknown" or not re.match(r"^[A-Za-z]+(?: [A-Za-z]+)*$", result):
        return "Unknown"
    return result


def validate_level(text: str) -> str:
    result = call_llm([{
        "role": "user",
        "content": f"""Extract the primary school level from: "{text}"
Rules:
- Return ONLY "Primary 1", "Primary 2", "Primary 3", "Primary 4", "Primary 5", or "Primary 6"
- If unclear, return "Unknown"

Examples:
"p5" → Primary 5
"primary 3" → Primary 3
"grade 4" → Unknown"""
    }])
    if not result or result not in ["Primary 1", "Primary 2", "Primary 3",
                                     "Primary 4", "Primary 5", "Primary 6"]:
        return "Unknown"
    return result


def validate_gender(text: str) -> str:
    result = call_llm([{
        "role": "user",
        "content": f"""Extract gender from: "{text}"
Rules:
- Return ONLY "Male", "Female", or "Unknown"

Examples:
"boy" → Male
"she" → Female
"my daughter" → Female
"idk" → Unknown"""
    }])
    if not result or result not in ["Male", "Female"]:
        return "Unknown"
    return result


def validate_phone(text: str) -> str:
    try:
        cleaned = re.sub(r"[\s\-\(\)\+]", "", text)
        if len(cleaned) > 8 and cleaned.startswith("65"):
            cleaned = cleaned[2:]
        if re.match(r"^\d{6,15}$", cleaned):
            return cleaned
        return "Unknown"
    except Exception:
        return "Unknown"


def validate_subject(text: str) -> str:
    result = call_llm([{
        "role": "user",
        "content": f"""Does this message indicate the person wants Math or Mathematics practice questions?

Message: "{text}"

Return ONLY "Math" if yes, or "Unknown" if no.

Examples:
"math" → Math
"mathematics" → Math
"i want math questions" → Math
"math mcq" → Math
"maths please" → Math
"science" → Unknown
"english" → Unknown
"hi" → Unknown"""
    }])
    if not result or result not in ["Math"]:
        return "Unknown"
    return result


def validate_difficulty(text: str) -> str:
    result = call_llm([{
        "role": "user",
        "content": f"""Extract difficulty level from: "{text}"
Rules:
- Return ONLY "Easy", "Medium", or "Hard"
- If unclear default to "Medium"

Examples:
"easy" → Easy
"hard questions" → Hard
"normal" → Medium
"medium" → Medium"""
    }])
    if not result or result not in ["Easy", "Medium", "Hard"]:
        return "Medium"
    return result


# ─────────────────────────────────────────
# REGISTRATION FLOW
# ─────────────────────────────────────────

def confirm_details(name, level, gender, phone) -> bool:
    print("\n📋 Please confirm your child's details:")
    print(f"   👦 Name:   {name}")
    print(f"   🎓 Level:  {level}")
    print(f"   👤 Gender: {gender}")
    print(f"   📱 Phone:  {phone}")
    return ask_yes_no("\nIs this correct? ")
    while True:
        try:
            confirm = safe_input("\nIs this correct? (yes/no): ").lower()
            if confirm in ["yes", "y"]:
                return True
            elif confirm in ["no", "n"]:
                return False
            else:
                print("⚠️  Please type 'yes' or 'no'.")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}. Please try again.")


def greet_and_register():
    global session, students_sheet

    print("Hi! Welcome to Whatsprep 👋")
    print("-" * 30)
    print("(You can type 'exit' at any time to quit)\n")

    students_sheet = get_sheets()

    # --- Grab their opening message ---
    opener = safe_input("You: ")

    # --- Try to extract name or student ID from opener ---
    identifier_type, identifier_value = extract_student_identifier(opener)

    if identifier_type in ["NAME", "ID"]:
        student = find_student_locally(identifier_type, identifier_value)
        if student:
            level_display = student.get("primary_level") or "Unknown level"
            if ask_yes_no(f"\nIs your child {student['name']} in {level_display}? "):
            
                session = {
                    "name": student["name"],
                    "level": student.get("primary_level", ""),
                    "gender": student.get("gender", ""),
                    "phone": student.get("phone", ""),
                    "student_id": student.get("student_id", "")
                }
                print(f"\n✅ Great! Let's get started, {session['name']}!")
                handle_menu()
                return
            else:
                print("\n🤔 No problem, let's verify via your phone number instead.")
        else:
            print("\n🤔 Couldn't find that student locally. Let's verify via phone.")

    # --- Phone lookup ---
    phone = ask_with_retries(
        "What is your WhatsApp number? ",
        validate_phone,
        "That doesn't look like a valid phone number. Please enter e.g. 81823031"
    )

    students_found = registry.check_student_exists(phone)
    if not students_found:
        students_found = check_existing_student_local(phone)

    if students_found:
        existing = pick_student(students_found)
        level_display = existing.get("primary_level") or "Unknown level"

        if ask_yes_no(f"\nIs your child {existing['name']} in {level_display}? "):
            session = {
                "name": existing["name"],
                "level": existing.get("primary_level", ""),
                "gender": existing.get("gender") or "",
                "phone": phone,
                "student_id": existing.get("student_id", "")
            }
            print(f"\n✅ Welcome back, {session['name']}!")
            handle_menu()
            return
        else:
            print("\n🤔 Let's register a new student under this number.")

    # --- New student registration ---
    name = ask_with_retries(
        "What is the student's name? ",
        validate_name,
        "I couldn't catch a valid name. Please write just the name e.g. 'Ayushi'."
    )
    level = ask_with_retries(
        "What is their schooling level? (e.g. P1, P2 ... P6): ",
        validate_level,
        "Please enter a level between Primary 1 and Primary 6 e.g. 'P4' or 'Primary 4'."
    )
    gender = ask_with_retries(
        "What is their gender? (e.g. boy, girl, he, she): ",
        validate_gender,
        "Please enter something like 'boy', 'girl', 'he', or 'she'."
    )

    while True:
        if confirm_details(name, level, gender, phone):
            break
        else:
            print("\n🔄 Which field would you like to fix?")
            print("   1. Name\n   2. Level\n   3. Gender\n   4. Phone")
            while True:
                field = safe_input("Enter the number (1-4): ")
                if field in ["1", "2", "3", "4"]:
                    break
                print("⚠️  Please enter a number between 1 and 4.")
            if field == "1":
                name = ask_with_retries("New name: ", validate_name,
                                        "I couldn't catch a valid name.")
            elif field == "2":
                level = ask_with_retries("New level (e.g. P3): ", validate_level,
                                         "Please enter Primary 1 to Primary 6.")
            elif field == "3":
                gender = ask_with_retries("New gender (boy/girl): ", validate_gender,
                                          "Please enter boy or girl.")
            elif field == "4":
                phone = ask_with_retries("New phone number: ", validate_phone,
                                         "That doesn't look like a valid phone number.")

    new_student = registry.register_student(name, level, gender, phone)
    student_id = (
        new_student.get("student_id", f"S{datetime.now().strftime('%Y%m%d%H%M%S')}")
        if new_student
        else f"S{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    if students_sheet:
        save_student_to_sheets(students_sheet, name, level, gender, phone)

    save_student_local(name, level, gender, phone, student_id)

    session = {
        "name": name,
        "level": level,
        "gender": gender,
        "phone": phone,
        "student_id": student_id
    }

    print("-" * 30)
    print(f"✅ {name} has been successfully registered in Whatsprep!")
    print(f"   🆔 Student ID: {student_id}")
    handle_menu()


# ─────────────────────────────────────────
# MAIN MENU — SUBJECT → TOPICS → SUBTOPICS → DIFFICULTY → GENERATE
# ─────────────────────────────────────────

def check_if_done(text: str) -> bool:
    result = call_llm([{
        "role": "user",
        "content": f"""Does this message mean the person wants to stop, end, or has nothing more to ask?

Message: "{text}"

Return ONLY "yes" or "no".

Rules:
- "no" on its own means YES they are done
- Only return "no" if they are clearly asking for math or a topic
- When in doubt about short replies like "no", "nope", "nothing", return "yes"

Examples:
"done" → yes
"bye" → yes
"no" → yes
"nope" → yes
"nothing else" → yes
"no thanks" → yes
"math mcq" → no
"math" → no
"yes please give me more" → no"""
    }])
    if not result:
        return False
    return result.lower() == "yes"


def ask_subject() -> str:
    print(f"\nWhat would you like practice questions for?")
    print("  We currently offer: Math")
    while True:
        user_input = safe_input("\nYou: ")
        if check_if_done(user_input):
            print(f"\n👋 Thank you for using Whatsprep, {session['name']}! Goodbye!")
            exit()
        subject = validate_subject(user_input)
        if subject != "Unknown":
            return subject
        print("⚠️  We currently only offer Math questions. Please type 'Math' to continue.")


def ask_topic(subject: str) -> tuple:
    topics = get_topics(session["level"], subject)
    if not topics:
        print(f"\n⚠️  Could not load topics. Please try again later.")
        return None, None

    print(f"\n📚 For {subject}, here are the available topics for {session['name']}:")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic.get('topic_name', f'Topic {i}')}")

    print("\nYou can select one or multiple topics e.g. '1' or '1,2,3' or '1 2 3'")

    while True:
        try:
            choice = safe_input("\nEnter the topic number(s): ")

            raw_choices = re.split(r"[,\s]+", choice.strip())
            indices = []
            valid = True

            for c in raw_choices:
                if not c:
                    continue
                try:
                    index = int(c) - 1
                    if 0 <= index < len(topics):
                        indices.append(index)
                    else:
                        print(f"⚠️  '{c}' is out of range. Please enter numbers between 1 and {len(topics)}.")
                        valid = False
                        break
                except ValueError:
                    print(f"⚠️  '{c}' is not a valid number.")
                    valid = False
                    break

            if not valid or not indices:
                continue

            # Remove duplicates while preserving order
            seen = set()
            indices = [i for i in indices if not (i in seen or seen.add(i))]

            selected_topics = [topics[i] for i in indices]
            topic_ids = [str(t.get("topic_id", "")) for t in selected_topics]
            topic_names = [t.get("topic_name", "") for t in selected_topics]

            print(f"\n✅ Selected topic(s):")
            for name in topic_names:
                print(f"   • {name}")

            return ",".join(topic_ids), ", ".join(topic_names)

        except ValueError:
            print("⚠️  Please enter valid numbers.")


def ask_subtopic(topic_id: str, topic_name: str) -> tuple:
    # Skip subtopic selection if multiple topics selected
    if "," in str(topic_id):
        print("📝 Multiple topics selected — using broad topics.")
        return None, None

    while True:
        choice = safe_input(
            f"\nWould you like to choose a specific subtopic under '{topic_name}', "
            f"or use the broad topic?\n"
            f"  1. Show me subtopics\n"
            f"  2. Use the broad topic\n"
            f"Enter (1/2): "
        )
        if choice in ["1", "2"]:
            break
        print("⚠️  Please enter 1 or 2.")

    if choice == "2":
        return None, None

    subtopics = get_subtopics(topic_id)
    if not subtopics:
        print("⚠️  Could not load subtopics. Using broad topic instead.")
        return None, None

    print(f"\n📖 Subtopics under '{topic_name}':")
    for i, sub in enumerate(subtopics, 1):
        print(f"  {i}. {sub.get('subtopic_name', sub.get('name', f'Subtopic {i}'))}")

    while True:
        try:
            pick = safe_input("\nEnter the number of the subtopic you want: ")
            index = int(pick) - 1
            if 0 <= index < len(subtopics):
                selected = subtopics[index]
                sub_id = selected.get("subtopic_id", selected.get("id", ""))
                sub_name = selected.get("subtopic_name", selected.get("name", ""))
                return sub_id, sub_name
            print(f"⚠️  Please enter a number between 1 and {len(subtopics)}.")
        except ValueError:
            print("⚠️  Please enter a number.")


def ask_difficulty() -> str:
    print("\nWhat difficulty would you like?")
    print("  1. Easy")
    print("  2. Medium")
    print("  3. Hard")
    while True:
        user_input = safe_input("\nYou: ")
        difficulty = validate_difficulty(user_input)
        if difficulty:
            return difficulty


def handle_menu():
    print(f"\nHow can I help you today, {session['name']}? 😊")
    print("-" * 30)
    print("Tell me what you'd like — e.g. 'I want Math questions' or 'Math mcq please'")
    print("-" * 30)

    while True:
        try:
            user_input = safe_input("\nYou: ")

            if not user_input:
                print("⚠️  Please type something so I can help you!")
                continue

            if check_if_done(user_input):
                print(f"\n👋 Thank you for using Whatsprep, {session['name']}! Goodbye!")
                exit()

            # --- Step 1: Subject ---
            subject = validate_subject(user_input)
            if subject == "Unknown":
                subject = ask_subject()

            # --- Step 2: Topics ---
            topic_id, topic_name = ask_topic(subject)
            if not topic_id:
                print("\nIs there anything else I can help you with?")
                continue

            # --- Step 3: Subtopics (optional) ---
            subtopic_id, subtopic_name = ask_subtopic(topic_id, topic_name)
            if subtopic_name:
                print(f"✅ Subtopic selected: {subtopic_name}")
            else:
                print(f"✅ Using broad topic(s): {topic_name}")

            # --- Step 4: Difficulty ---
            difficulty = ask_difficulty()
            print(f"✅ Difficulty: {difficulty}")

            # --- Step 5: Generate assessment(s) ---
            final_topic_id = subtopic_id if subtopic_id else topic_id

            print(f"\n⏳ Generating your worksheet now...")
            print(f"   📚 Subject:    {subject}")
            print(f"   📝 Topic(s):   {topic_name}")
            if subtopic_name:
                print(f"   🔖 Subtopic:   {subtopic_name}")
            print(f"   ⚡ Difficulty: {difficulty}")
            print(f"   🎓 Level:      {session['level']}")

            # generate-assessment only accepts ONE topic_id — handle multi-topic selection
            topic_id_list = str(final_topic_id).split(",")

            if len(topic_id_list) > 1:
                print(f"\n⏳ Generating {len(topic_id_list)} assessments (one per topic)...")
                urls = []
                for single_topic_id in topic_id_list:
                    url = generate_worksheet_url(
                        topic_id=single_topic_id.strip(),
                        level=session["level"],
                        subject=subject,
                        difficulty=difficulty,
                        student_id=str(session.get("student_id", ""))
                    )
                    if url:
                        urls.append(url)

                if urls:
                    print(f"\n✅ Your worksheets are ready!")
                    for i, u in enumerate(urls, 1):
                        print(f"   🔗 Worksheet {i}: {u}")
                else:
                    print("\n🚧 Worksheet generation coming soon!")
            else:
                url = generate_worksheet_url(
                    topic_id=topic_id_list[0].strip(),
                    level=session["level"],
                    subject=subject,
                    difficulty=difficulty,
                    student_id=str(session.get("student_id", ""))
                )

                if url:
                    print(f"\n✅ Your worksheet is ready!")
                    print(f"   🔗 {url}")
                else:
                    print("\n🚧 Worksheet generation coming soon!")

            print("\nIs there anything else I can help you with?")

        except KeyboardInterrupt:
            print(f"\n\n👋 Thank you for using Whatsprep, {session['name']}! Goodbye!")
            exit()
        except Exception as e:
            print(f"⚠️  Something went wrong: {e}. Please try again.")
            continue

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
greet_and_register()

