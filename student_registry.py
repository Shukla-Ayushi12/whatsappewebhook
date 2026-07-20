# student_registry.py
import requests


class StudentRegistry:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "jwt-token": api_key,
            "Content-Type": "application/json"
        }

    def check_student_exists(self, phone: str) -> list | None:
        """Check if student(s) exist using POST /user-details-by-contact."""
        formatted_phone = f"+65{phone}" if not phone.startswith("+") else phone
        print(f"🔍 Checking if {formatted_phone} exists on platform...")

        try:
            payload = {"mobile": formatted_phone}
            response = requests.post(
                f"{self.base_url}/user-details-by-contact",
                headers=self.headers,
                json=payload,
                timeout=10
            )

            print(f"📡 Status: {response.status_code}")
            print(f"📡 Response: {response.text}")

            if "<!DOCTYPE html>" in response.text:
                print("⚠️  Wrong endpoint. Falling back to local database.")
                return None

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, dict):
                    records = data.get("data", [])
                    total = data.get("total_records", 0)

                    if total == 0 or not records:
                        print("✅ Student not found — new registration needed.")
                        return None

                    return self._parse_students(records, phone)

                elif isinstance(data, list):
                    if not data:
                        return None
                    return self._parse_students(data, phone)

            elif response.status_code == 404:
                print("✅ Student not found — new registration needed.")
                return None

            else:
                print(f"⚠️  Unexpected response {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            print("⚠️  Could not connect. Falling back to local database.")
            return None
        except requests.exceptions.Timeout:
            print("⚠️  Request timed out. Falling back to local database.")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            return None

    def _parse_students(self, records: list, phone: str) -> list:
        """Parse student records from API response into normalized dicts."""
        result = []
        for s in records:
            # API returns level inside "class" (single object) or "classes" (array)
            class_obj = s.get("class", {})
            classes_arr = s.get("classes", [])

            if class_obj and isinstance(class_obj, dict):
                level = class_obj.get("level_name", "")
            elif classes_arr and isinstance(classes_arr, list) and len(classes_arr) > 0:
                level = classes_arr[0].get("level_name", "")
            else:
                level = s.get("level", "")

            result.append({
                "name": s.get("name", "Unknown"),
                "primary_level": level,
                "gender": s.get("gender") or "",
                "phone": phone,
                "student_id": s.get("id", "")
            })
        return result

    def register_student(self, name: str, level: str, gender: str,
                         phone: str) -> dict | None:
        """Register a new student on the platform."""
        formatted_phone = f"+65{phone}" if not phone.startswith("+") else phone
        print(f"📤 Registering {name} on platform...")

        try:
            payload = {
                "name": name,
                "mobile": formatted_phone,
                "level": level,
                "gender": gender.lower()
            }

            print(f"📦 Payload: {payload}")

            response = requests.post(
                f"{self.base_url}/create-student",
                headers=self.headers,
                json=payload,
                timeout=10
            )

            print(f"📡 Status: {response.status_code}")
            print(f"📡 Response: {response.text}")

            if "<!DOCTYPE html>" in response.text:
                print("⚠️  Wrong endpoint. Saving locally instead.")
                return None

            if response.status_code in [200, 201]:
                data = response.json()
                student = data.get("data", {})
                print("✅ Student registered successfully on platform!")
                return {
                    "student_id": student.get("id", ""),
                    "name": student.get("name", name),
                    "full_name": student.get("full_name", ""),
                    "phone": student.get("phone_number", phone),
                    "gender": student.get("gender") or "",
                    "primary_level": level,
                    "created_at": student.get("created_at", "")
                }
            elif response.status_code == 409:
                print("⚠️  Student already exists on platform.")
                return None
            elif response.status_code == 400:
                print(f"⚠️  Bad request: {response.text}")
                return None
            elif response.status_code == 401:
                print("⚠️  Unauthorized. Please check your API key.")
                return None
            else:
                print(f"⚠️  Unexpected response {response.status_code}: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print("⚠️  Could not connect. Saving locally instead.")
            return None
        except requests.exceptions.Timeout:
            print("⚠️  Request timed out. Saving locally instead.")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error registering student: {e}")
            return None

    def update_student(self, student_id: str, updates: dict) -> bool:
        """Update an existing student's details."""
        print(f"📝 Updating student {student_id}...")

        try:
            response = requests.put(
                f"{self.base_url}/students/{student_id}",
                headers=self.headers,
                json=updates,
                timeout=10
            )

            print(f"📡 Status: {response.status_code}")

            if response.status_code in [200, 204]:
                print("✅ Student updated successfully!")
                return True
            else:
                print(f"⚠️  Could not update student: {response.status_code}")
                return False

        except Exception as e:
            print(f"⚠️  Unexpected error updating student: {e}")
            return False

            
