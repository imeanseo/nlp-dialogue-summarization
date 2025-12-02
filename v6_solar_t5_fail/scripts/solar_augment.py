import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os
import re
from collections import Counter, defaultdict
import concurrent.futures


def clean_dialogue(dialogue_str):
    lines = dialogue_str.strip().split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('---'):
            cleaned.append(line)
    return ' '.join(cleaned)


def detect_speech_style(dialogue):
    dialogue = dialogue.lower()
    honorific_count = len(re.findall(r'[해|합니다|세요|주|드|시|ㅂ니다]', dialogue))
    plain_count = len(re.findall(r'해|한다|야|지|네|구나', dialogue))
    if honorific_count > plain_count * 1.5:
        return "존댓말 (-습니다/합니다 체)"
    else:
        return "반말 (한다/해 체)"


def extract_real_examples(df, n_examples_per_topic=2):
    examples = {}
    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic].head(10)
        topic_samples = topic_df.sample(n=min(n_examples_per_topic, len(topic_df)), random_state=42)
        topic_examples = []
        for _, row in topic_samples.iterrows():
            dialogue = clean_dialogue(row['dialogue'])
            summary = row['summary']
            style = detect_speech_style(dialogue)
            example = f"""예시 ({style}):
대화: {dialogue[:200]}...
요약: {summary}"""
            topic_examples.append(example)
        examples[topic] = "\n".join(topic_examples)
    return examples


def solar_real_data_fewshot_prompt(dialogue, topic, real_examples):
    style = detect_speech_style(dialogue)
    base_rules = f"""다음 대화의 핵심만 1-2문장으로 요약해주세요.

📋 필수 규칙:
1. "#Person1#이 #Person2#에게 ..." 형식 반드시 사용
2. 문체: {style} (대화 따라 자동 적용)
3. 핵심만! (등장인물+행동+결과, 50-100자)
4. 디테일/대화 인용/반복 표현 제외"""

    topic_examples = real_examples.get(topic, "")
    return f"""{base_rules}

📚 {topic} 주제 실제 예시:
{topic_examples}

🎯 이번 대화:
{dialogue}

요약:"""


API_KEY = "secret" 
API_URL = "https://api.upstage.ai/v1/chat/completions"
MODEL_NAME = "solar-pro2"


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
        "max_tokens": 100,
        "stream": False,
    }
    try:
        resp = requests.post(API_URL, json=data, headers=headers, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"요청 오류: {e}")
        return "요약 생성 실패"


def augment_sample(row, real_examples):
    prompt = solar_real_data_fewshot_prompt(row['dialogue'], row['topic'], real_examples)
    summary = call_solar(prompt)
    return {
        'dialogue': row['dialogue'],
        'summary': summary,
        'topic': row['topic'],
        'source': 'solar_real_fewshot'
    }


def balanced_topic_augmentation(df, min_samples_per_topic=10, augment_ratio=3):
    print("🔍 실제 train.csv에서 토픽별 Few-shot 예시 추출 중...")
    real_examples = extract_real_examples(df)

    df['dialogue'] = df['dialogue'].apply(clean_dialogue)
    topic_counts = df['topic'].value_counts()

    print("\n📊 원본 토픽 분포 (상위 10):")
    print(topic_counts.head(10))

    # 🚀 필터링: 샘플 3개 이상 토픽만 선택
    filtered_topics = topic_counts[topic_counts >= 3].index
    print(f"\n🔍 증강 대상 토픽: {len(filtered_topics)}개 (샘플 3개 이상)")

    results = []
    save_path = "data/augmented/train_solar_real_fewshot_partial.csv"
    os.makedirs("data/augmented", exist_ok=True)

    for topic in tqdm(filtered_topics, desc="토픽별 증강"):
        topic_df = df[df['topic'] == topic]
        current_count = len(topic_df)

        # 원본 데이터 전부 포함
        for _, row in topic_df.iterrows():
            results.append({
                'dialogue': row['dialogue'],
                'summary': row['summary'],
                'topic': topic,
                'source': 'original'
            })

        # 10개 미만이면 10개로 증강
        target_count = min_samples_per_topic
        if current_count < target_count:
            need_augment = target_count - current_count
            print(f"\n🎯 {topic}: {current_count}→{target_count} (+{need_augment})")

            augment_samples = topic_df.sample(
                min(need_augment * augment_ratio, len(topic_df)),
                random_state=42
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(augment_sample, row, real_examples) for _, row in augment_samples.iterrows()]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    if i % 10 == 0:
                        pd.DataFrame(results).to_csv(save_path, index=False)

    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\n✅ 중간 결과를 {save_path} 에 저장했습니다.")
    return pd.DataFrame(results), real_examples


if __name__ == "__main__":
    print("🚀 V6 실제 데이터 기반 토픽별 Few-shot 증강")
    print("🎯 목표: 토픽당 최소 10개 + 문체 정확 반영")

    df = pd.read_csv("/root/nlp_data/train.csv")
    print(f"📂 원본 데이터 로드: {len(df)} 샘플")

    aug_df, real_examples = balanced_topic_augmentation(df, min_samples_per_topic=10)

    final_save_path = "data/augmented/train_solar_real_fewshot.csv"
    aug_df.to_csv(final_save_path, index=False)

    with open("data/augmented/real_fewshot_examples.json", "w", encoding="utf-8") as f:
        json.dump(real_examples, f, ensure_ascii=False, indent=2)

    print("\n✅ 증강 완료!")
    print(f"총 샘플: {len(aug_df)}")
    print("\n📊 증강 후 균형 (하위 10 토픽):")
    print(aug_df.groupby('topic').size().sort_values().head(10))
    print("\n📊 증강 후 균형 (상위 10 토픽):")
    print(aug_df.groupby('topic').size().sort_values(ascending=False).head(10))

    print("\n🔥 Solar 증강 샘플 5개:")
    display_df = aug_df[aug_df['source']=='solar_real_fewshot'][['topic', 'summary']].head(5)
    for _, row in display_df.iterrows():
        print(f"[{row['topic']}] {row['summary']}")

    # 최종 학습 데이터 (원본 + 증강)
    orig_train = pd.read_csv("/root/nlp_data/train.csv")
    orig_train['source'] = 'original_full'
    solar_only = aug_df[aug_df['source']=='solar_real_fewshot']
    v6_train = pd.concat([orig_train, solar_only])
    v6_train.to_csv("data/augmented/train_v6_perfect.csv", index=False)
    print(f"\n🎉 V6 최종 학습 데이터: {len(v6_train)} 샘플 (원본+증강)")
