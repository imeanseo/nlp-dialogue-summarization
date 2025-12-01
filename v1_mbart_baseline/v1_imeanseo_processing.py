# ========================================
# v1_processing.py
# 전처리 + Config 생성 (WandB 제외)
# ========================================

import pandas as pd
import numpy as np
import re
import os
import json
import yaml
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("v1_processing.py - 데이터 전처리")
print("=" * 60)

# ========================================
# 1. Config 설정 (학습용)
# ========================================

config_data = {
    'general': {
        'data_path': './',
        'model_name': 'facebook/mbart-large-50-many-to-many-mmt',
        'output_dir': './checkpoints',
    },
    
    'tokenizer': {
        'encoder_max_len': 400,
        'decoder_max_len': 80,
        'special_tokens': [],  # 나중에 자동으로 채워짐
    },
    
    'training': {
        'overwrite_output_dir': True,
        'num_train_epochs': 5,
        'learning_rate': 3e-5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'lr_scheduler_type': 'cosine',
        'optim': 'adamw_torch',
        'gradient_accumulation_steps': 2,
        'save_steps': 500,
        'eval_steps': 500,
        'save_total_limit': 3,
        'fp16': True,
        'load_best_model_at_end': True,
        'seed': 42,
        'logging_dir': './logs',
        'logging_strategy': 'steps',
        'logging_steps': 100,
        'predict_with_generate': True,
        'generation_max_length': 80,
        'do_train': True,
        'do_eval': True,
        'early_stopping_patience': 3,
        'early_stopping_threshold': 0.001,
        'report_to': 'wandb',
    },
    
    'wandb': {
        'entity': 'imeanseo_',  # 수정!
        'project': 'dialogue-summarization',
        'name': 'v1-mbart-baseline',
    },
    
    'inference': {
        'no_repeat_ngram_size': 2,
        'early_stopping': True,
        'generate_max_length': 80,
        'num_beams': 4,
        'batch_size': 32,
        'remove_tokens': ['<usr>', '</s>', '<s>', '<pad>'],
    },
}

print("✅ Config 설정 완료")

# ========================================
# 2. 데이터 로드
# ========================================

print("\n📂 데이터 로딩...")

train_df = pd.read_csv('train.csv')
dev_df = pd.read_csv('dev.csv')
test_df = pd.read_csv('test.csv')

print(f"✅ Train: {len(train_df):,}개")
print(f"✅ Dev:   {len(dev_df):,}개")
print(f"✅ Test:  {len(test_df):,}개")

# ========================================
# 3. 전처리 함수
# ========================================

print("\n🔧 전처리 함수 정의...")

def clean_text(text):
    """텍스트 정리"""
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def postprocess_text(text, remove_tokens):

    if pd.isna(text):
        return ""
    
    # Remove tokens
    for token in remove_tokens:
        text = text.replace(token, '')
    
    # Person + 조사 띄어쓰기 제거
    text = re.sub(
        r'(#Person\d+#)\s+(은|는|이|가|을|를|에게|께서|과|와|의|도|만|부터|까지|에|에서)',
        r'\1\2',
        text
    )
    
    text = text.replace('\t', '').replace('  ', ' ')
    text = text.strip()
    return text

# 전처리 적용
print("\n🧹 전처리 적용 중...")

train_df['dialogue_clean'] = train_df['dialogue'].apply(clean_text)
train_df['summary_clean'] = train_df['summary'].apply(clean_text)
dev_df['dialogue_clean'] = dev_df['dialogue'].apply(clean_text)
dev_df['summary_clean'] = dev_df['summary'].apply(clean_text)
test_df['dialogue_clean'] = test_df['dialogue'].apply(clean_text)

print("✅ 전처리 완료")

# 샘플 확인
print("\n📝 전처리 샘플:")
print(f"[원본] {train_df['dialogue'].iloc[0][:100]}...")
print(f"[처리] {train_df['dialogue_clean'].iloc[0][:100]}...")

# ========================================
# 4. Special Tokens 추출
# ========================================

print("\n🔧 Special Tokens 추출...")

def extract_special_tokens(dataframe):
    pattern = r'#\w+#'
    all_text = ' '.join(dataframe['dialogue'].astype(str))
    tokens = re.findall(pattern, all_text)
    return sorted(list(set(tokens)))

special_tokens = extract_special_tokens(train_df)

print(f"✅ 발견: {len(special_tokens)}개")
for i, token in enumerate(special_tokens[:10], 1):
    print(f"  {i}. {token}")

# Config에 저장
config_data['tokenizer']['special_tokens'] = special_tokens

# ========================================
# 5. 토크나이저로 실제 길이 확인
# ========================================

print(f"\n🔤 토크나이저 로드...")

tokenizer = AutoTokenizer.from_pretrained(
    config_data['general']['model_name'],
    src_lang="ko_KR",
    tgt_lang="ko_KR"
)

# Special Token 추가
num_added = tokenizer.add_special_tokens({
    'additional_special_tokens': special_tokens
})
print(f"✅ {num_added}개 Special Token 추가")

# 샘플링
print("\n📊 토큰 길이 분석 (샘플 1000개)...")
sample_size = 1000
sample_indices = np.random.choice(len(train_df), min(sample_size, len(train_df)), replace=False)

dialogue_tokens = []
summary_tokens = []

for idx in sample_indices:
    d_tokens = tokenizer(train_df['dialogue_clean'].iloc[idx], truncation=False)
    dialogue_tokens.append(len(d_tokens['input_ids']))
    
    s_tokens = tokenizer(train_df['summary_clean'].iloc[idx], truncation=False)
    summary_tokens.append(len(s_tokens['input_ids']))

dialogue_tokens = np.array(dialogue_tokens)
summary_tokens = np.array(summary_tokens)

# 통계
print("\n[대화문 토큰 길이]")
print(f"  평균:     {dialogue_tokens.mean():.1f}")
print(f"  중간값:   {np.median(dialogue_tokens):.0f}")
print(f"  95%:      {np.percentile(dialogue_tokens, 95):.0f}")
print(f"  최대:     {dialogue_tokens.max()}")

print("\n[요약문 토큰 길이]")
print(f"  평균:     {summary_tokens.mean():.1f}")
print(f"  중간값:   {np.median(summary_tokens):.0f}")
print(f"  95%:      {np.percentile(summary_tokens, 95):.0f}")
print(f"  최대:     {summary_tokens.max()}")

print(f"\n📋 Config 설정값:")
print(f"  encoder_max_len: {config_data['tokenizer']['encoder_max_len']}")
print(f"  decoder_max_len: {config_data['tokenizer']['decoder_max_len']}")

# ========================================
# 6. 저장
# ========================================

print("\n💾 저장 중...")

os.makedirs('./processed_data', exist_ok=True)

# 전처리된 데이터
train_df.to_csv('./processed_data/train_processed.csv', index=False)
dev_df.to_csv('./processed_data/dev_processed.csv', index=False)
test_df.to_csv('./processed_data/test_processed.csv', index=False)

# Config 저장 (YAML & JSON 둘 다)
with open('./config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

with open('./processed_data/config.json', 'w', encoding='utf-8') as f:
    json.dump(config_data, f, ensure_ascii=False, indent=2)

# Special tokens 별도 저장
with open('./processed_data/special_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(special_tokens, f, ensure_ascii=False, indent=2)

print("✅ 저장 완료:")
print("  - ./config.yaml")
print("  - ./processed_data/train_processed.csv")
print("  - ./processed_data/dev_processed.csv")
print("  - ./processed_data/test_processed.csv")
print("  - ./processed_data/config.json")
print("  - ./processed_data/special_tokens.json")

# ========================================
# 7. 시각화
# ========================================

print("\n📊 시각화 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 대화문
axes[0].hist(dialogue_tokens, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].axvline(dialogue_tokens.mean(), color='red', linestyle='--', 
                label=f'Mean: {dialogue_tokens.mean():.0f}')
axes[0].axvline(config_data['tokenizer']['encoder_max_len'], 
                color='orange', linestyle='--', linewidth=2,
                label=f"Config: {config_data['tokenizer']['encoder_max_len']}")
axes[0].set_title('Dialogue Token Length Distribution')
axes[0].set_xlabel('Token Length')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 요약문
axes[1].hist(summary_tokens, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].axvline(summary_tokens.mean(), color='red', linestyle='--',
                label=f'Mean: {summary_tokens.mean():.0f}')
axes[1].axvline(config_data['tokenizer']['decoder_max_len'],
                color='orange', linestyle='--', linewidth=2,
                label=f"Config: {config_data['tokenizer']['decoder_max_len']}")
axes[1].set_title('Summary Token Length Distribution')
axes[1].set_xlabel('Token Length')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./processed_data/token_distribution.png', dpi=300, bbox_inches='tight')
print("✅ 시각화 저장: ./processed_data/token_distribution.png")
plt.close()

# ========================================
# 완료
# ========================================

print("\n" + "=" * 60)
print("✅ 전처리 완료!")
print("=" * 60)
print("\n📁 생성된 파일:")
print("  1. config.yaml               - 학습 설정 (학습 시 사용)")
print("  2. processed_data/           - 전처리된 데이터")
print("  3. token_distribution.png    - 시각화")
