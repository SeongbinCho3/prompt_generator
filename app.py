"""
app.py — Flask 백엔드
OpenAI API를 통해 프롬프트를 생성하는 서버
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# ── .env 파일 로드 ──────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)  # 프런트엔드(127.0.0.1:5500 등)에서의 요청 허용

# ── OpenAI 클라이언트 초기화 ────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── APE 기법 기반 시스템 프롬프트 ──────────────────────────
# 28번 APE(Automatic Prompt Engineer) 기법을 적용:
# 후보 생성 → 자체 평가 → 교차 최적화 → 최적 프롬프트 출력
SYSTEM_PROMPT = """
당신은 프롬프트 엔지니어링 전문가입니다.
사용자가 제공한 정보를 바탕으로 APE(Automatic Prompt Engineer) 기법을 적용해
ChatGPT, Gemini, Claude 등 어떤 AI에서도 잘 작동하는 최적화된 프롬프트를 생성합니다.

[APE 적용 절차]
STEP 1 — 후보 프롬프트 3개 내부 생성 (출력하지 않음)
  각 후보는 서로 다른 전략 사용:
  - 역할 부여형: "당신은 XXX 전문가입니다"
  - 단계별 지시형: "STEP 1: ... STEP 2: ..."
  - 제약 명시형: "반드시 ~하되, ~하지 마세요"

STEP 2 — 각 후보를 아래 기준으로 내부 평가 (출력하지 않음)
  - 명확성 (25점): 지시가 모호하지 않은가
  - 완전성 (25점): 필요한 정보가 모두 있는가
  - 실행가능성 (25점): AI가 바로 실행 가능한가
  - 품질유도력 (25점): 높은 품질의 결과를 이끌어내는가

STEP 3 — 상위 후보들의 장점을 결합해 최종 최적 프롬프트 1개 생성

STEP 4 — 최적 프롬프트만 출력
  - 완성된 프롬프트 본문만 출력
  - 설명, 평가 과정, 부연 내용은 일절 출력하지 않음
  - 사용자가 AI에 그대로 붙여넣어 즉시 사용 가능한 형태
  - 한국어로 작성 (단, 영문 프롬프트 요청 시 영문 작성)

[참고할 프롬프트 기법 목록]
- 서술형 / 지침형 / 함수형
- 제로샷 / 원샷 / 퓨샷
- 페르소나 패턴 (역할·경력·전문분야·말투 포함)
- 마크다운 구조화 (#헤더, >인용, |표, 코드블록)
- 표현 강도 조절 (권장→일반→강조→절대)
- 톤 지정 (전문적/친근/유머러스/감성적 등)
- 대안 접근법 패턴 (단순나열/비교/상황분기/역발상)
- 이용자 페르소나 패턴
- 레시피 패턴 (단계별 절차)
- 뒤집힌 상호작용 패턴 (AI가 먼저 질문)
- 인지 검증자 패턴 (하위 질문 분해 후 통합)
- 질문 개선 패턴
- 팩트체크 목록 패턴
- 메타언어 생성 패턴 (단축 명령어)
- 리플렉션 패턴 (초안→비평→개선)
- 아웃라인 확장 패턴
- 컨텍스트 관리자 패턴
- 무한 생성 패턴 (변수 조합)
- 5W1H / CO-STAR / FOCUS / ROSES / RISEN / BAB 프레임워크
- 다중 관점 기법
- 심사숙고 유도 기법 (Chain of Thought)
- APE (Automatic Prompt Engineer)
"""


# ── /chat 엔드포인트 ────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data:
        return jsonify({"error": "요청 데이터가 없습니다."}), 400

    user_prompt = data.get("userPrompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "userPrompt 필드가 필요합니다."}), 400

    # 프런트엔드에서 전달된 추가 시스템 프롬프트가 있으면 병합
    extra_system = data.get("systemPrompt", "")
    final_system = SYSTEM_PROMPT
    if extra_system:
        final_system += f"\n\n[추가 지침]\n{extra_system}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",          # 필요시 gpt-4o-mini로 변경 가능
            max_tokens=1500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": final_system},
                {"role": "user",   "content": user_prompt}
            ]
        )

        result = response.choices[0].message.content
        return jsonify({"result": result})

    except Exception as e:
        print(f"[OpenAI 오류] {e}")
        return jsonify({"error": str(e)}), 500


# ── 서버 실행 ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
