"""Diagnostic v2: inspect full response, try larger tokens + disable thinking."""
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

c = genai.Client()

PROMPT = "Score this clinical note. CANDIDATE: 'Patient is 50yo male with hypertension.' REFERENCE: 'Patient is 50yo female with diabetes.'"
SYS = 'Output STRICT JSON only: {"factuality": int 1-5, "completeness": int 1-5, "format": int 1-5, "rationale": "..."}'

def try_call(label, **cfg):
    print(f"\n========== {label} ==========")
    try:
        r = c.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT,
            config=types.GenerateContentConfig(
                system_instruction=SYS,
                temperature=0,
                response_mime_type="application/json",
                **cfg,
            ),
        )
        print("text repr:", repr(r.text)[:300])
        print("finish_reason:", getattr(r.candidates[0], "finish_reason", "?") if r.candidates else "(no candidate)")
        if r.usage_metadata:
            um = r.usage_metadata
            print(f"usage: prompt={um.prompt_token_count} "
                  f"candidates={um.candidates_token_count} "
                  f"thoughts={getattr(um, 'thoughts_token_count', '?')} "
                  f"total={um.total_token_count}")
    except Exception as e:
        print(f"EXCEPTION: {e}")

# A. 기본 (실패한 케이스)
try_call("A. max_tokens=300 (현재 설정)", max_output_tokens=300)

# B. max_tokens 크게
try_call("B. max_tokens=2000", max_output_tokens=2000)

# C. thinking 비활성화
try_call("C. max_tokens=300 + thinking_budget=0",
         max_output_tokens=300,
         thinking_config=types.ThinkingConfig(thinking_budget=0))
