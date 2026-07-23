"""
Local test harness for the WhatsPrep webhook.

Talks straight to handle() so you can exercise the whole conversation flow
in a terminal, with no Meta, no tunnel, and no deploy.

Run:  python test_local.py
Quit: Ctrl+C
"""

from main import handle, SESSIONS

# Pretend to be one parent. Change this to test a different number.
PHONE = "6591234567"


def main():
    print("WhatsPrep local test. Type a message and press enter.")
    print("Commands: /reset clears the session, /state shows it.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not text:
            continue

        if text == "/reset":
            SESSIONS.pop(PHONE, None)
            print("[session cleared]\n")
            continue

        if text == "/state":
            print(f"[{SESSIONS.get(PHONE)}]\n")
            continue

        try:
            reply = handle(PHONE, text)
        except Exception as e:
            print(f"[handler error] {type(e).__name__}: {e}\n")
            continue

        print(f"\nBot: {reply}\n")
        print(f"[step: {SESSIONS.get(PHONE, {}).get('step')}]\n")


if __name__ == "__main__":
    main()

    