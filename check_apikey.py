# --- OpenAI 연결 빠른 자가진단 (붙여넣고 실행) ---
import os, json, urllib.request, urllib.error

def openai_self_test():
    key  =""
    base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    proj = ""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not key:
        print("✗ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    # 1) 인증 핑(모델 리스트) — 키/프로젝트/조직 헤더 확인
    req = urllib.request.Request(f"{base}/v1/models")
    req.add_header("Authorization", f"Bearer {key}")
    if proj: req.add_header("OpenAI-Project", proj)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("✓ Auth OK:", r.status)
    except urllib.error.HTTPError as e:
        print("✗ Auth 실패:", e.code)
        try: print(e.read().decode("utf-8"))
        except: pass
        return

    # 2) ChatCompletions 핑 — 정확히 PONG만 응답하도록 테스트
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with exactly the word: PONG"},
            {"role": "user", "content": "ping"}
        ],
        "temperature": 0
    }
    req2 = urllib.request.Request(f"{base}/v1/chat/completions",
                                  data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                  method="POST")
    req2.add_header("Authorization", f"Bearer {key}")
    req2.add_header("Content-Type", "application/json")
    if proj: req2.add_header("OpenAI-Project", proj)
    try:
        with urllib.request.urlopen(req2, timeout=30) as r:
            js = json.loads(r.read().decode("utf-8"))
            out = js["choices"][0]["message"]["content"].strip()
            print("✓ Chat OK:", out)
            if out != "PONG":
                print("  (참고) 응답이 정확히 'PONG'이 아니면 시스템 프롬프트가 무시된 것일 수 있어요.")
    except urllib.error.HTTPError as e:
        print("✗ Chat 실패:", e.code)
        try: print(e.read().decode("utf-8"))
        except: pass

openai_self_test()
