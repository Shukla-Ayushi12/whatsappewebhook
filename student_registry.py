import re
import logging
import requests

log = logging.getLogger("whatsprep.registry")

LEVELS = ["Primary 1", "Primary 2", "Primary 3",
          "Primary 4", "Primary 5", "Primary 6"]

_LEVEL_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def normalize_level(raw) -> str:
    """Map whatever the platform returns for a level onto 'Primary N'.

    Handles "P4", "p4", "Pri 4", "Primary Four", "Level 4", bare "4".
    Returns "" for anything outside primary, so the caller can decide what
    to do rather than guessing.
    """
    if not raw:
        return ""
    t = str(raw).strip().lower()
    if re.search(r"secondary|\bsec\b|\bjc|junior college", t):
        return ""
    for word, digit in _LEVEL_WORDS.items():
        t = re.sub(rf"\b{word}\b", str(digit), t)
    m = re.search(r"([1-6])(?!\d)", t)
    if not m:
        log.warning("Unrecognised level from platform: %r", raw)
        return ""
    return f"Primary {m.group(1)}"


class StudentRegistry:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "jwt-token": api_key,
            "Content-Type": "application/json",
        }
        self.last_error: str | None = None

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _format_phone(phone: str) -> str:
        return f"+65{phone}" if not str(phone).startswith("+") else str(phone)

    @staticmethod
    def _normalize_gender(gender: str) -> str:
        g = str(gender or "").strip().lower()
        if g in ("male", "boy", "guy", "m"):
            return "male"
        if g in ("female", "girl", "f"):
            return "female"
        return g

    def _classify(self, response) -> str | None:
        """Map a response onto a last_error value. None means it's usable."""
        if "<!DOCTYPE html>" in response.text[:200]:
            log.warning("Platform returned HTML, check base_url: %s", self.base_url)
            return "bad_endpoint"
        if response.status_code == 401:
            log.error("Platform rejected the token (401). JWT may have expired.")
            return "auth"
        if response.status_code == 400:
            log.warning("Bad request: %s", response.text[:300])
            return "bad_request"
        if response.status_code == 409:
            return "exists"
        if response.status_code in (200, 201, 204, 404):
            return None
        log.warning("Unexpected status %s: %s",
                    response.status_code, response.text[:300])
        return "unexpected"

    # ------------------------------------------------------------ lookup
    def check_student_exists(self, phone: str) -> list | None:
        """Find students registered to a phone number.

        Returns a list of normalised student dicts, or None.

        None is ambiguous on its own, so always check `last_error`:
            last_error is None  -> genuinely no student for this number
            last_error is set   -> the lookup failed, do NOT register anyone
        """
        self.last_error = None
        formatted = self._format_phone(phone)
        log.info("Looking up %s on platform", formatted)

        try:
            response = requests.post(
                f"{self.base_url}/user-details-by-contact",
                headers=self.headers,
                json={"mobile": formatted},
                timeout=10,
            )
        except requests.exceptions.Timeout:
            log.warning("Lookup timed out for %s", formatted)
            self.last_error = "timeout"
            return None
        except requests.exceptions.ConnectionError:
            log.warning("Could not connect to platform for %s", formatted)
            self.last_error = "unreachable"
            return None
        except Exception:
            log.exception("Lookup failed for %s", formatted)
            self.last_error = "error"
            return None

        log.info("Lookup status %s", response.status_code)

        problem = self._classify(response)
        if problem:
            self.last_error = problem
            return None

        if response.status_code == 404:
            log.info("No student found for %s", formatted)
            return None

        try:
            data = response.json()
        except ValueError:
            log.warning("Lookup returned non-JSON: %s", response.text[:200])
            self.last_error = "bad_endpoint"
            return None

        if isinstance(data, dict):
            records = data.get("data") or []
            if not records or data.get("total_records", len(records)) == 0:
                log.info("No student found for %s", formatted)
                return None
            return self._parse_students(records, phone)

        if isinstance(data, list):
            if not data:
                log.info("No student found for %s", formatted)
                return None
            return self._parse_students(data, phone)

        log.warning("Unexpected payload shape: %s", type(data).__name__)
        self.last_error = "unexpected"
        return None

    def _parse_students(self, records: list, phone: str) -> list:
        """Normalise platform records. Level comes from `class` or `classes`."""
        result = []
        for s in records:
            class_obj = s.get("class") or {}
            classes_arr = s.get("classes") or []

            if isinstance(class_obj, dict) and class_obj.get("level_name"):
                raw_level = class_obj.get("level_name", "")
            elif isinstance(classes_arr, list) and classes_arr:
                raw_level = classes_arr[0].get("level_name", "")
            else:
                raw_level = s.get("level", "")

            level = normalize_level(raw_level)
            if raw_level and not level:
                log.warning("Student %s has a level we can't map: %r",
                            s.get("id"), raw_level)

            first_name = (s.get("first_name") or "").strip()
            last_name = (s.get("last_name") or "").strip()
            
            full_name = s.get("name") or f"{first_name} {last_name}".strip() or "Unknown"

            result.append({
                "student_id": s.get("id", ""),
                "first_name": first_name,
                "last_name": last_name,
                "name": full_name,
                "primary_level": level,
                "raw_level": raw_level,
                "gender": s.get("gender") or "",
                "phone": phone,
            })
        log.info("Parsed %d student record(s)", len(result))
        return result

    # ------------------------------------------------------------ create
    def register_student(self, first_name: str, last_name: str, level: str,
                         gender: str, phone: str) -> dict | None:
        """Register a new student using first name and last name.

        Returns the created student dict, or None. On None, check
        `last_error`: "exists" means a 409, so the caller should re-run
        check_student_exists and adopt the existing record.
        """
        self.last_error = None
        formatted = self._format_phone(phone)
        level = normalize_level(level) or level

        clean_first = first_name.strip()
        clean_last = last_name.strip()
        clean_gender = self._normalize_gender(gender)
        full_name = f"{clean_first} {clean_last}".strip()

        log.info("Registering %s (%s) on platform", full_name, level)

        payload = {
            "first_name": clean_first,
            "last_name": clean_last,
            "mobile": formatted,
            "level": level,
            "gender": clean_gender,
        }

        try:
            response = requests.post(
                f"{self.base_url}/create-student",
                headers=self.headers,
                json=payload,
                timeout=10,
            )
            log.info("Create student platform response (%s): %s", response.status_code, response.text[:300])
        except requests.exceptions.Timeout:
            log.warning("Registration timed out for %s", full_name)
            self.last_error = "timeout"
            return None
        except requests.exceptions.ConnectionError:
            log.warning("Could not connect to platform to register %s", full_name)
            self.last_error = "unreachable"
            return None
        except Exception:
            log.exception("Registration failed for %s", full_name)
            self.last_error = "error"
            return None

        log.info("Registration status %s", response.status_code)

        problem = self._classify(response)
        if problem:
            self.last_error = problem
            if problem == "exists":
                log.info("%s already exists on the platform", full_name)
            return None

        if response.status_code not in (200, 201):
            self.last_error = "unexpected"
            return None

        try:
            student = (response.json() or {}).get("data") or {}
        except ValueError:
            log.warning("Registration returned non-JSON: %s", response.text[:200])
            self.last_error = "bad_endpoint"
            return None

        student_id = student.get("id", "")
        if not student_id:
            log.warning("Registration succeeded but returned no id: %s",
                        response.text[:200])
            self.last_error = "unexpected"
            return None

        returned_first = student.get("first_name") or clean_first
        returned_last = student.get("last_name") or clean_last
        returned_full = student.get("name") or f"{returned_first} {returned_last}".strip()

        log.info("Registered %s as %s", returned_full, student_id)
        return {
            "student_id": student_id,
            "first_name": returned_first,
            "last_name": returned_last,
            "name": returned_full,
            "phone": student.get("phone_number", phone),
            "gender": student.get("gender") or clean_gender,
            "primary_level": normalize_level(level),
            "raw_level": level,
            "created_at": student.get("created_at", ""),
        }

    # ------------------------------------------------------------ update
    def update_student(self, student_id: str, updates: dict) -> bool:
        self.last_error = None
        log.info("Updating student %s", student_id)
        try:
            response = requests.put(
                f"{self.base_url}/students/{student_id}",
                headers=self.headers,
                json=updates,
                timeout=10,
            )
        except requests.exceptions.Timeout:
            self.last_error = "timeout"
            return False
        except requests.exceptions.ConnectionError:
            self.last_error = "unreachable"
            return False
        except Exception:
            log.exception("Update failed for %s", student_id)
            self.last_error = "error"
            return False

        problem = self._classify(response)
        if problem:
            self.last_error = problem
            return False
        if response.status_code in (200, 204):
            log.info("Updated student %s", student_id)
            return True
        self.last_error = "unexpected"
        return False


if __name__ == "__main__":
    cases = {
        "Primary 4": "Primary 4", "P4": "Primary 4", "p4": "Primary 4",
        "PRIMARY 4": "Primary 4", "Pri 4": "Primary 4", "4": "Primary 4",
        "Primary Four": "Primary 4", "primary four": "Primary 4",
        "Grade 4": "Primary 4", "P 4": "Primary 4", "Level 4": "Primary 4",
        "P1": "Primary 1", "P6": "Primary 6",
        "Secondary 2": "", "Sec 1": "", "JC1": "", "JC 2": "",
        "": "", None: "", "Primary 10": "",
    }
    bad = 0
    for raw, expected in cases.items():
        got = normalize_level(raw)
        if got != expected:
            bad += 1
            print(f"MISMATCH {raw!r}: got {got!r}, expected {expected!r}")
    print("normalize_level: all pass" if not bad else f"{bad} failing")
