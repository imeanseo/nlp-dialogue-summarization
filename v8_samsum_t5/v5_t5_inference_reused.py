import os
import yaml
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("=" * 80)
print("v5_inference.py - T5-Large 추론")
print("=" * 80)

# Config 로드
with open('./v5_t5_large/v5_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"\n✅ Config 로드 완료")

# 테스트 모드 설정
TEST_MODE = False # False로 바꾸면 전체 500개 처리
TEST_SAMPLES = 10

# 모델 & Tokenizer 로드
model_path = os.path.join(config['general']['output_dir'], "final_model")
print(f"\n🤖 모델 로드 중...")
print(f"  경로: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"  Device: {device}")

# 테스트 데이터 로드
print(f"\n📂 테스트 데이터 로드 중...")
test_df = pd.read_csv(os.path.join(config['general']['data_path'], 'test.csv'))

if TEST_MODE:
    test_df = test_df.head(TEST_SAMPLES)
    print(f"⚠️ 테스트 모드: {TEST_SAMPLES}개만 추론")

print(f"✅ Test 데이터: {len(test_df)}개")

# 추론 함수
def generate_summary(dialogue, max_length=120, num_beams=4):
    """
    T5로 요약 생성
    """
    # T5 prefix 추가
    input_text = "summarize: " + dialogue
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=config['tokenizer']['encoder_max_len'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
    
    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# 추론 실행
print(f"\n🔮 추론 시작...")
summaries = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
    dialogue = row['dialogue']
    summary = generate_summary(dialogue)
    summaries.append(summary)
    
    # 처음 3개 출력
    if idx < 3:
        print(f"\n[{idx}] {row['fname']}")
        print(f"  대화: {dialogue[:100]}...")
        print(f"  요약: {summary}")


from datetime import datetime
import pytz  # 시간대 라이브러리

# 결과 저장
kst = pytz.timezone('Asia/Seoul')
date_str = datetime.now(kst).strftime('%m%d_%H%M') 
output_filename = f"submission_samsum_{'test' if TEST_MODE else 'full'}_{date_str}.csv"
submission = pd.DataFrame({
    'fname': test_df['fname'],
    'summary': summaries
})
submission.to_csv(output_filename, index=False, encoding='utf-8-sig')
os.makedirs('./predictions', exist_ok=True)
submission.to_csv(f'./predictions/{output_filename}', index=False)

print(f"\n✅ 추론 완료!")
print(f"  저장 파일: {output_filename}")
print(f"  총 {len(summaries)}개 요약 생성")

# 최종 확인
print(f"\n📋 제출 파일 미리보기:")
print(submission.head(10))
