#!/usr/bin/env python3
"""
Run this to verify the web app and chat API work before telling users to run.
Usage: python verify_web.py
Exits 0 if all checks pass, 1 otherwise.
"""
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"


def main():
    os.chdir(PROJECT_DIR)
    # Start server in background
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        env={**os.environ, "PORT": str(PORT)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=PROJECT_DIR,
    )
    try:
        # Wait for server to respond
        for _ in range(25):
            try:
                with urllib.request.urlopen(BASE + "/", timeout=2) as r:
                    if r.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        else:
            print("FAIL: Server did not start in time", file=sys.stderr)
            return 1

        # GET /
        with urllib.request.urlopen(BASE + "/", timeout=5) as r:
            if r.status != 200:
                print(f"FAIL: GET / returned {r.status}", file=sys.stderr)
                return 1
            data = r.read().decode()
            if "chat-widget" not in data or "Julie" not in data:
                print("FAIL: Page missing chat widget or Julie", file=sys.stderr)
                return 1

        # POST /chat
        req = urllib.request.Request(
            BASE + "/chat",
            data=json.dumps({"message": "hello"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            out = json.loads(r.read().decode())
            if "reply" not in out or "end_session" not in out:
                print("FAIL: /chat response missing reply or end_session", file=sys.stderr)
                return 1

        print("OK: All checks passed. Web app is ready.")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
