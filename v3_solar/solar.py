import random
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================
# 1. 기본 설정
# ============================
import requests

API_KEY = ""  # 실제 키로 교체
API_URL = "https://api.upstage.ai/v1/chat/completions"
MODEL_NAME = "solar-pro2"

N_SAMPLES = 500  # 증강 개수 대폭 축소
INPUT_PATH = "./train.csv"  # 원본만 (스왑 없음)
AUG_PATH = "./train_solar_dev_style_aug.csv"
FULL_PATH = "./train_solar_dev_style_full.csv"

def call_solar(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    }

    resp = requests.post(API_URL, json=data, headers=headers, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["choices"][0]["message"]["content"].strip()


N_SAMPLES = 1000                              # 증강에 쓸 샘플 개수
INPUT_PATH = "./train_augmented.csv"
AUG_PATH = "./train_solar_fewshot_aug.csv"
FULL_PATH = "./train_solar_fewshot_full.csv"


FEW_SHOT_EXAMPLES_FROM_DEV = """
[예시 1 - 대화]
#Person1#: 안녕하세요, 오늘 기분이 어떠세요?
#Person2#: 요즘 숨쉬기가 힘들어요.
#Person1#: 최근에 감기에 걸렸나요?
#Person2#: 아니요, 감기는 안 걸렸어요. 숨쉴 때 가슴이 답답해요.
#Person1#: 혹시 알고 있는 알레르기 있으세요?
#Person2#: 아니요, 특별히 알고 있는 알레르기는 없어요.
#Person1#: 이게 항상 그런가요, 아니면 주로 활동할 때 그런가요?
#Person2#: 운동할 때 특히 많이 그래요.
#Person1#: 천식 검사를 위해 폐 전문의에게 가보시는 게 좋겠어요.
#Person2#: 도와주셔서 감사합니다, 의사 선생님.

[예시 1 - 요약]
#Person2#는 숨쉬기 어려워합니다. 의사는 #Person2#에게 증상을 확인하고, 천식 검사를 위해 폐 전문의에게 가볼 것을 권합니다.


[예시 2 - 대화]
#Person1#: 실례합니다, Mr. White? 제가 가기 전에 이거 사인 좀 해주셔야 해요.
#Person2#: 알겠어요, Sherry. 기다리게 해서 미안해요. 네가 말해주지 않았으면 아마 깜빡했을 거예요.
#Person1#: 그게 제 일이죠, 선생님. 여기 한 번 더 사인 부탁드려요.
#Person2#: 여기 있어요.

[예시 2 - 요약]
Sherry가 Mr. White에게 사인을 요청합니다.


[예시 3 - 대화]
#Person1#: 여보, 담배 좀 끊는 게 좋을 것 같아.
#Person2#: 왜? 나 담배 피울 때 멋지다고 했잖아.
#Person1#: 그런데 건강을 위해서라도 그만 뒀으면 좋겠어.
#Person2#: 담배가 해로운 건 알아.
#Person1#: 이 기사 좀 봐봐. 담배가 폐암을 유발할 수 있다고 해.
#Person2#: 안 믿어.
#Person1#: 그래도 담배가 건강에 해롭다는 건 알지?
#Person2#: 당연히 알지, 그런데 담배 끊는 거 쉽지 않은 거 알잖아...
#Person1#: 말 돌리지 마. 담배 끊을 거야, 안 끊을 거야?
#Person2#: 네, 여사님. 말씀대로 하죠.

[예시 3 - 요약]
#Person1#은 #Person2#에게 건강을 위해 담배를 끊을 것을 권유하고, #Person2#은 이를 어렵다고 느끼지만 동의합니다.


[예시 4 - 대화]
#Person1#: 2번 게이트가 어디 있어요? 세 시간 동안 이 공항 안에 있었는데 못 찾겠어요.
#Person2#: 가까이 있어요. 계단을 올라가서 왼쪽으로 가세요.
#Person1#: 올라가서 왼쪽이요?
#Person2#: 네, 거기 간판이 있어요. 공항이 더 작으면 좋겠어요!

[예시 4 - 요약]
#Person1#은 2번 게이트를 찾지 못하고, #Person2#는 위층 왼쪽으로 가라고 안내합니다.
"""
# ============================
# solar 호출 함수
# ============================

def call_solar(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False,
    }
    resp = requests.post(API_URL, json=data, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def solar_augment_dev_style(dialogue: str, base_summary: str) -> str:
    prompt = (
        "너는 한국어 대화 요약을 잘하는 모델이야.\n"
        "아래 예시들의 스타일을 정확히 따라서, 새로운 대화를 요약해 줘.\n\n"
        "**중요 규칙:**\n"
        "1. #Person1#, #Person2# 같은 태그가 대화에 있으면 절대 바꾸지 말고 그대로 사용할 것.\n"
        "2. 대화에서 실제 이름(예: Sherry, Mr. White), 장소, 시간 등이 나오면 그것을 우선 사용.\n"
        "3. 요약은 2~3문장으로, 핵심 행동과 이유만 간결하게.\n"
        "4. 불필요한 설명, 리스트, 괄호 코멘트는 절대 쓰지 말 것.\n"
        "5. 동사는 '~합니다' 또는 '~한다' 형태로 통일. 대화문의 형태가 반말의 형태로 이루어진 경우는 ~한다, 존댓말로 대화가 이루어지는 경우 ~합니다를 사용할 것\n\n"
        f"{FEW_SHOT_EXAMPLES_FROM_DEV}\n\n"
        "[새로운 대화]\n"
        f"{dialogue}\n\n"
        "[새로운 요약]"
    )
    return call_solar(prompt)


# ============================
# 메인 로직
# ============================

def main():
    print(f"📂 원본 train 로드: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    if len(df) <= N_SAMPLES:
        subset = df.copy()
        print(f"  데이터가 {len(df)}개라 전체를 증강합니다.")
    else:
        subset = df.sample(N_SAMPLES, random_state=42)
        print(f"  전체 {len(df)}개 중 {N_SAMPLES}개 샘플링하여 증강합니다.")
    
    aug_rows = []
    
    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Solar dev-style 증강"):
        dialogue = str(row["dialogue"])
        base_summary = str(row["summary"])
        
        try:
            new_summary = solar_augment_dev_style(dialogue, base_summary)
        except Exception as e:
            print(f"\n⚠️ Solar 호출 에러 (idx={idx} fname={row.get('fname','')}): {e}")
            continue
        
        aug_rows.append({
            "fname": f'{row["fname"]}_soldev',
            "dialogue": dialogue,
            "summary": new_summary,
            "topic": row.get("topic", ""),
        })
    
    if not aug_rows:
        print("⚠️ 생성된 증강 데이터가 없습니다.")
        return
    
    aug_df = pd.DataFrame(aug_rows)
    print(f"\n✅ 증강 데이터 개수: {len(aug_df)}")
    
    # 증강 데이터만 저장
    aug_df.to_csv(AUG_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 증강 데이터 저장: {AUG_PATH}")
    
    # 원본 + 증강 합치기
    full_df = pd.concat([df, aug_df], ignore_index=True)
    print(f"✅ 합쳐진 전체 데이터 개수: {len(full_df)}")
    
    full_df.to_csv(FULL_PATH, index=False, encoding="utf-8-sig")
    print(f"💾 전체 데이터 저장: {FULL_PATH}")


if __name__ == "__main__":
    main()
