import requests
import pandas as pd
from tqdm import tqdm

FEW_SHOT_TEMPLATE = """대화를 한 문장으로 간결하게 요약하세요. 구체적인 숫자, 시간, 가격은 생략하고 핵심 행동만 작성합니다.

### 예시:
대화:
#Person1#: 안녕하세요, Mr. Smith. 저는 Dr. Hawkins입니다. 오늘 무슨 일로 오셨어요?
#Person2#: 건강검진을 받으려고 왔어요.
#Person1#: 네, 5년 동안 검진을 안 받으셨네요. 매년 한 번씩 받으셔야 해요.
요약: Mr. Smith는 Dr. Hawkins에게 건강검진을 받으러 와서 매년 검진 필요성을 안내받았습니다.

대화:
#Person1#: 저기요, 열쇠 세트 본 적 있어요?
#Person2#: 어떤 종류의 열쇠요?
#Person1#: 열쇠 다섯 개랑 작은 발 장식이 달려 있어요.
요약: #Person1#은 열쇠 세트를 잃어버리고 #Person2#에게 찾는 것을 도와달라고 요청합니다.

대화:
{dialogue}
요약:"""

def generate_summary(dialogue):
    prompt = FEW_SHOT_TEMPLATE.format(dialogue=dialogue)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "exaone3.5:7.8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 80,
                    "stop": ["\n", "대화:", "###"]
                }
            },
            timeout=120
        )
        return response.json()["response"].strip()
    except:
        return "요약문입니다."

test_df = pd.read_csv("test.csv")
print("🚀 EXAONE 추론 시작!")

summaries = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    summary = generate_summary(row['dialogue'])
    summaries.append(summary)

pd.DataFrame({"fname": test_df['fname'], "summary": summaries}).to_csv(
    "submission_exaone.csv", index=False, encoding='utf-8-sig'
)
print("✅ submission_exaone.csv 생성 완료!")
