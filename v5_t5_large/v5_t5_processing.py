import os
import yaml
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

print("=" * 80)
print("v5_processing.py - T5-Large 데이터 전처리")
print("=" * 80)

# Config 로드
with open('v5_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"\n✅ Config 로드 완료")
print(f"  Model: {config['general']['model_name']}")
print(f"  Encoder Max Length: {config['tokenizer']['encoder_max_len']}")
print(f"  Decoder Max Length: {config['tokenizer']['decoder_max_len']}")

# 데이터 로드 (클리닝된 파일 사용!)
print(f"\n📂 데이터 로드 중...")
train_df = pd.read_csv(os.path.join(config['general']['data_path'], 'train_cleaned.csv'))
dev_df = pd.read_csv(os.path.join(config['general']['data_path'], 'dev.csv'))
test_df = pd.read_csv(os.path.join(config['general']['data_path'], 'test.csv'))

print(f"  Train: {len(train_df)}개")
print(f"  Dev: {len(dev_df)}개")
print(f"  Test: {len(test_df)}개")

# Tokenizer 로드
print(f"\n🔧 Tokenizer 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    config['general']['model_name'],
    cache_dir=config['general'].get('cache_dir', None)
)
print(f"  Vocab size: {tokenizer.vocab_size}")

# T5는 prefix를 사용함
# 예: "summarize: <대화문>" → "<요약문>"
def preprocess_function(examples):
    """
    T5 전용 전처리
    - Input: "summarize: <dialogue>"
    - Target: "<summary>"
    """
    # T5는 task prefix를 사용
    inputs = ["summarize: " + doc for doc in examples['dialogue']]
    targets = examples['summary']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config['tokenizer']['encoder_max_len'],
        padding='max_length',
        truncation=True
    )
    
    # Tokenize targets (labels)
    labels = tokenizer(
        targets,
        max_length=config['tokenizer']['decoder_max_len'],
        padding='max_length',
        truncation=True
    )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# Dataset 변환
print(f"\n🔄 Dataset 변환 중...")
train_dataset = Dataset.from_pandas(train_df[['dialogue', 'summary']])
eval_dataset = Dataset.from_pandas(dev_df[['dialogue', 'summary']])
test_dataset = Dataset.from_pandas(test_df[['dialogue']])

# Tokenization 적용
print(f"\n⚙️ Tokenization 적용 중...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['dialogue', 'summary'],
    desc="Tokenizing train"
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['dialogue', 'summary'],
    desc="Tokenizing dev"
)

# Test는 labels 없음
def preprocess_test(examples):
    inputs = ["summarize: " + doc for doc in examples['dialogue']]
    return tokenizer(
        inputs,
        max_length=config['tokenizer']['encoder_max_len'],
        padding='max_length',
        truncation=True
    )

test_dataset = test_dataset.map(
    preprocess_test,
    batched=True,
    remove_columns=['dialogue'],
    desc="Tokenizing test"
)

# 저장
output_dir = os.path.join(config['general']['data_path'], 'processed_data_v5')
os.makedirs(output_dir, exist_ok=True)

print(f"\n💾 저장 중...")
train_dataset.save_to_disk(os.path.join(output_dir, 'train'))
eval_dataset.save_to_disk(os.path.join(output_dir, 'eval'))
test_dataset.save_to_disk(os.path.join(output_dir, 'test'))

print(f"\n✅ 전처리 완료!")
print(f"  저장 위치: {output_dir}/")
print(f"  - train: {len(train_dataset)}개")
print(f"  - eval: {len(eval_dataset)}개")
print(f"  - test: {len(test_dataset)}개")

# 샘플 확인
print(f"\n📝 샘플 확인 (첫 번째 데이터):")
sample = train_dataset[0]
print(f"  Input IDs 길이: {len(sample['input_ids'])}")
print(f"  Labels 길이: {len(sample['labels'])}")
print(f"  실제 텍스트 (디코딩):")
print(f"    Input: {tokenizer.decode(sample['input_ids'][:100])}...")
print(f"    Label: {tokenizer.decode([id for id in sample['labels'] if id != -100][:50])}...")
