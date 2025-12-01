# ========================================
# v3_processing.py (v3.2 수정판)
# ========================================

import pandas as pd
import yaml
import os
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("v3_processing.py - LLM 데이터셋 생성 (v3.2)")
print("=" * 60)

# 1. Config 로드
with open('./v3_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 2. 데이터 로드
print("\n📂 데이터 로딩...")
train_df = pd.read_csv('./train.csv')
dev_df = pd.read_csv('./dev.csv')
test_df = pd.read_csv('./test_fixed.csv')

print(f"Train: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")
print(f"Test:  {len(test_df)}")

# 3. 프롬프트 템플릿
template = config['tokenizer']['prompt_template']

def format_instruction(row, is_test=False):
    """
    v3.2: 프롬프트 생성 개선
    """
    # 1) 대화문 전처리
    dialogue = str(row['dialogue']).strip()
    
    # 2) 템플릿 적용
    prompt = template.format(dialogue=dialogue)
    
    # 3) 학습용
    if not is_test:
        summary = str(row['summary']).strip()
        # EOS 토큰 확실히 추가
        full_text = f"{prompt}{summary}</s>"
        return full_text
    else:
        # ★★★ 테스트용: 공백/줄바꿈 없이 깔끔하게 ★★★
        return prompt

# 4. 데이터 변환
print("\n🔄 프롬프트 적용 중...")

train_df['text'] = train_df.apply(lambda x: format_instruction(x, is_test=False), axis=1)
dev_df['text'] = dev_df.apply(lambda x: format_instruction(x, is_test=False), axis=1)
test_df['prompt'] = test_df.apply(lambda x: format_instruction(x, is_test=True), axis=1)

# 5. 저장
os.makedirs('./processed_data_v3', exist_ok=True)
train_df.to_csv('./processed_data_v3/train.csv', index=False)
dev_df.to_csv('./processed_data_v3/dev.csv', index=False)
test_df.to_csv('./processed_data_v3/test.csv', index=False)

# 6. 검증
print("\n✅ 생성 완료! 검증 중...")

# 학습 데이터 검증
sample_train = train_df['text'].iloc[0]
print("\n📝 학습 데이터 샘플 (끝부분):")
print(sample_train[-150:])
print(f"\n✅ EOS 토큰 포함: {'</s>' in sample_train}")

# 테스트 데이터 검증
sample_test = test_df['prompt'].iloc[0]
print("\n📝 테스트 프롬프트 샘플 (끝부분):")
print(sample_test[-150:])
print(f"\n프롬프트 끝: {repr(sample_test[-50:])}")

print("\n✅ v3 데이터셋 생성 완료! (./processed_data_v3/)")
