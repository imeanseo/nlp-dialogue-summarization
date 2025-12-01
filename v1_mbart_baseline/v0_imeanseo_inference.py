# ========================================
# inference.py
# mBART Inference & Submission 생성
# ========================================

import pandas as pd
import numpy as np
import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("v2_inference.py - Inference & Submission")
print("=" * 60)

# ========================================
# 1. Config 불러오기
# ========================================

print("\n📖 Config 불러오기...")
config_path = './v2_config.yaml'

with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

print("✅ Config 로드 완료")

# ========================================
# 2. GPU 확인
# ========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ========================================
# 3. 모델 & 토크나이저 로드
# ========================================

print("\n🤖 Best 모델 로드...")

model_path = os.path.join(config['general']['output_dir'], 'best_model')

# 모델 경로 확인
if not os.path.exists(model_path):
    print(f"⚠️ Best model 없음: {model_path}")
    # 최신 체크포인트 찾기
    import glob
    checkpoints = glob.glob(os.path.join(config['general']['output_dir'], "checkpoint-*"))
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        model_path = checkpoints[-1]
        print(f"📥 최신 체크포인트 사용: {model_path}")
    else:
        print("❌ 체크포인트가 없습니다!")
        exit(1)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    src_lang='ko_KR',
    tgt_lang='ko_KR'
)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to(device)
model.eval()

print(f"✅ 모델 로드 완료: {model_path}")


# ========================================
# 4. 테스트 데이터 로드
# ========================================

print("\n📂 테스트 데이터 로딩...")

# 수정된 test_fixed.csv 사용
import os
if os.path.exists('./test_fixed.csv'):
    test_df = pd.read_csv('./test_fixed.csv')
    print(f"✅ Test (fixed): {len(test_df):,}개")
else:
    test_df = pd.read_csv('./test.csv')
    print(f"✅ Test (original): {len(test_df):,}개")

# fname 확인
print(f"   fname 범위: {test_df['fname'].iloc[0]} ~ {test_df['fname'].iloc[-1]}")

# fname 순서대로 정렬
test_df['fname_num'] = test_df['fname'].str.extract('(\d+)').astype(int)
test_df = test_df.sort_values('fname_num').reset_index(drop=True)

# 전처리
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

test_df['dialogue_clean'] = test_df['dialogue'].apply(clean_text)

print(f"✅ 전처리 완료: {len(test_df):,}개")

# ========================================
# 5. Inference 함수
# ========================================

print("\n🔮 Inference 함수 준비...")

def generate_summary(dialogue, model, tokenizer, device, config):
    """단일 대화문 요약 생성"""

    # 토크나이징
    inputs = tokenizer(
        dialogue,
        max_length=config['tokenizer']['encoder_max_len'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config['tokenizer']['decoder_max_len'],
            num_beams=config['inference']['num_beams'],
            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
            early_stopping=config['inference']['early_stopping'],
        )

    # 디코딩
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 후처리: remove_tokens 제거
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        summary = summary.replace(token, '')

    # fix_missing_subjects 호출하여 주어 누락 보완
    summary = fix_missing_subjects(summary)

    # 최종 공백 등 정리
    summary = summary.replace('\t', '').replace('  ', ' ').strip()

    return summary

print("✅ Inference 함수 준비 완료")

import re

def fix_missing_subjects(summary: str) -> str:
    """정교한 주어 보완 - 과도한 반복 방지"""

    # 0) 연속된 #Person 태그 축약
    summary = re.sub(r'(#Person\d+#)+', r'#Person1#', summary)

    # 1) 문장 시작이 조사(은/는/이/가/에게/과/와 등)로 시작하면 #Person1# 붙이기
    if re.match(r'^[은는이이가을를에게과와]', summary):
        summary = '#Person1#' + summary

    # 2) 마침표/느낌표/물음표 뒤에 조사만 오는 경우 → #Person1# 붙이기
    summary = re.sub(
        r'([.!?])\s*([은는이이가을를에게과와])',
        r'\1 #Person1#\2',
        summary
    )

    # 3) " , 가/는/은/을" 패턴 보완
    # 앞 문장에 #Person1#/ #Person2#가 있으면, 없던 쪽을 채워 넣기
    if "가 " in summary or "는 " in summary or "은 " in summary or "을 " in summary:
        if "#Person1#" in summary and "#Person2#" not in summary:
            summary = summary.replace(" 가 ", " #Person1#가 ")
            summary = summary.replace(" 는 ", " #Person1#는 ")
            summary = summary.replace(" 은 ", " #Person1#은 ")
            summary = summary.replace(" 을 ", " #Person1#을 ")
        elif "#Person2#" in summary and "#Person1#" not in summary:
            summary = summary.replace(" 가 ", " #Person2#가 ")
            summary = summary.replace(" 는 ", " #Person2#는 ")
            summary = summary.replace(" 은 ", " #Person2#은 ")
            summary = summary.replace(" 을 ", " #Person2#을 ")

    # 4) "의 ~" 앞에 주어 보완 (대부분 #Person2#)
    summary = summary.replace(" 의 아파트", " #Person2#의 아파트")
    summary = summary.replace(" 의 가방", " #Person2#의 가방")
    summary = summary.replace(" 의 집", " #Person2#의 집")
    summary = summary.replace(" 의 자동차", " #Person2#의 자동차")
    summary = summary.replace(" 의 휴대폰", " #Person2#의 휴대폰")
    summary = summary.replace(" 의 가게", " #Person2#의 가게")
    summary = summary.replace(" 의 청구서", " #Person2#의 청구서")
    summary = summary.replace(" 의 차", " #Person2#의 차")
    summary = summary.replace(" 의 방문", " #Person2#의 방문")

    # 5) “과/와 는” 패턴 → 뒤 사람을 #Person2#로 가정
    # 예: "#Person1#과 는" → "#Person1#과 #Person2#는"
    summary = re.sub(
        r'(#Person1#)\s*(과|와)\s+는',
        r'\1\2 #Person2#는',
        summary
    )

    # 6) "에게" 앞에 아무 것도 없거나 공백만 있는 경우 → #Person2#에게
    summary = re.sub(
        r'\s에게',
        ' #Person2#에게',
        summary
    )
    summary = re.sub(
        r'^에게',
        '#Person2#에게',
        summary
    )

    # 7) "#Person1#과 #Person1#" → "#Person1#과 #Person2#"
    summary = summary.replace("#Person1#과 #Person1#", "#Person1#과 #Person2#")
    summary = summary.replace("#Person1#와 #Person1#", "#Person1#와 #Person2#")

    # 8) "#Person1#은 #Person1#가 / 는 / 을 ..." 패턴 교정
    summary = summary.replace("#Person1#은 #Person1#가", "#Person1#은 #Person2#가")
    summary = summary.replace("#Person1#은 #Person1#는", "#Person1#은 #Person2#는")
    summary = summary.replace("#Person1#은 #Person1#을", "#Person1#은 #Person2#을")
    

    # 2) 너무 장황한 괄호/대괄호 제거 (혹시 남아 있다면)
    summary = re.sub(r'\(.*?\)', '', summary)
    summary = re.sub(r'\[.*?\]', '', summary)

    # 9) Person + 조사 사이 띄어쓰기 정리
    summary = re.sub(
        r'(#Person\d+#)\s+(은|는|이|가|을|를|에게|께서|과|와|의|도|만|부터|까지|에|에서)',
        r'\1\2',
        summary
    )

    # 10) 공백 정리
    summary = re.sub(r'\s+', ' ', summary).strip()

    return summary


# ========================================
# 6. 배치 Inference (완전 수정 버전)
# ========================================

print("\n🚀 Inference 시작...")
print(f"  Total: {len(test_df):,}개")
print(f"  Batch Size: {config['inference']['batch_size']}")
print(f"  Num Beams: {config['inference']['num_beams']}")

remove_tokens = config['inference']['remove_tokens']  # 변수 정의
summaries = []

batch_size = config['inference']['batch_size']

for i in tqdm(range(0, len(test_df), batch_size), desc="Generating"):
    batch = test_df['dialogue_clean'].iloc[i:i+batch_size].tolist()
    
    # 토크나이징
    inputs = tokenizer(
        batch,
        max_length=config['tokenizer']['encoder_max_len'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config['tokenizer']['decoder_max_len'],
            num_beams=config['inference']['num_beams'],
            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
            early_stopping=config['inference']['early_stopping'],
        )
    
    # 디코딩
    batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 후처리 (완전 수정!)
    for summary in batch_summaries:
        # 1. remove_tokens 제거
        for token in remove_tokens:
            summary = summary.replace(token, '')
        
        # 2. 주어 보완 (핵심!)
        summary = fix_missing_subjects(summary)
        
        # 3. 최종 정리
        summary = re.sub(r'\s+', ' ', summary).replace('\t', '').strip()
        summaries.append(summary)

print(f"✅ Inference 완료: {len(summaries)}개 생성")


# ========================================
# 7. 제출 파일 생성
# ========================================

print("\n💾 제출 파일 생성...")

# 버전 정보
version = 'v2'
model_name = 'mbart'
date_str = datetime.now().strftime('%m%d')  # MMDD 형식

# 파일명 생성
filename = f'submission_{version}_{model_name}_baseline_{date_str}.csv'

# sample_submission 형식
submission = pd.DataFrame({
    'fname': [f'test_{i}' for i in range(len(summaries))],
    'summary': summaries
})

# 저장
os.makedirs('./predictions', exist_ok=True)
submission_path = os.path.join('./predictions', filename)
submission.to_csv(submission_path, index=False)

print(f"✅ 제출 파일 저장: {submission_path}")
print(f"   파일명: {filename}")


# ========================================
# 8. 샘플 확인
# ========================================

print("\n📝 생성 샘플 (처음 3개):")
print("=" * 60)

for i in range(min(3, len(test_df))):
    print(f"\n[{i+1}번째 샘플]")
    print(f"대화문: {test_df['dialogue_clean'].iloc[i][:100]}...")
    print(f"요약문: {summaries[i]}")
    print("-" * 60)

# ========================================
# 9. 통계
# ========================================

print("\n📊 생성 요약문 통계:")
summary_lengths = [len(s) for s in summaries]
print(f"  평균 길이: {np.mean(summary_lengths):.1f}자")
print(f"  최소 길이: {np.min(summary_lengths)}자")
print(f"  최대 길이: {np.max(summary_lengths)}자")

# ========================================
# 완료
# ========================================

print("\n" + "=" * 60)
print("✅ inference.py 완료!")
print("=" * 60)
print(f"\n📁 생성된 파일:")
print(f"  - {submission_path}")
print(f"\n🚀 다음 단계:")
print(f"  1. 제출 파일 확인: cat {submission_path} | head")
print(f"  2. 대회 사이트에 제출!")
print("=" * 60)

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("\n🧹 GPU 메모리 정리 완료")
