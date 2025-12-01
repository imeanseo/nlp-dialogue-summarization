# ========================================
# v4_processing.py
# Llama-3 Korean 데이터셋 생성
# ========================================

import pandas as pd
import yaml
import os
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("v4_processing.py - Llama-3 데이터셋 생성")
print("=" * 60)

# 1. Config 로드
with open('./v4_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 2. 데이터 로드
print("\n📂 데이터 로딩...")
train_df = pd.read_csv('./train.csv')
dev_df = pd.read_csv('./dev.csv')
test_df = pd.read_csv('./test_fixed.csv')

print(f"Train: {len(train_df)}")
print(f"Dev:   {len(dev_df)}")
print(f"Test:  {len(test_df)}")

# 3. 간단한 프롬프트 템플릿
template = config['tokenizer']['prompt_template']

def format_instruction(row, is_test=False):
    """
    v4: 간단한 프롬프트 (Llama용)
    """
    dialogue = str(row['dialogue']).strip()
    prompt = template.format(dialogue=dialogue)
    
    if not is_test:
        summary = str(row['summary']).strip()
        return f"{prompt}{summary}</s>"
    else:
        return prompt

# 4. 데이터 변환
print("\n🔄 프롬프트 적용 중...")

train_df['text'] = train_df.apply(lambda x: format_instruction(x, is_test=False), axis=1)
dev_df['text'] = dev_df.apply(lambda x: format_instruction(x, is_test=False), axis=1)
test_df['prompt'] = test_df.apply(lambda x: format_instruction(x, is_test=True), axis=1)

# 5. 저장
os.makedirs('./processed_data_v4', exist_ok=True)
train_df.to_csv('./processed_data_v4/train.csv', index=False)
dev_df.to_csv('./processed_data_v4/dev.csv', index=False)
test_df.to_csv('./processed_data_v4/test.csv', index=False)

# 6. 검증
print("\n✅ 생성 완료! 검증 중...")

sample_train = train_df['text'].iloc[0]
print("\n📝 학습 데이터 샘플 (끝부분):")
print(sample_train[-200:])
print(f"\n✅ EOS 토큰 포함: {'</s>' in sample_train}")

sample_test = test_df['prompt'].iloc[0]
print("\n📝 테스트 프롬프트 샘플 (끝부분):")
print(sample_test[-150:])

print("\n✅ v4 데이터셋 생성 완료! (./processed_data_v4/)")
