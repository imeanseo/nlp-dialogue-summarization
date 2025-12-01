import os
import yaml
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

print("=" * 80)
print("v5_train.py - T5-Large 학습")
print("=" * 80)

# Config 로드
with open('v5_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"\n✅ Config 로드 완료")
print(f"  Model: {config['general']['model_name']}")

# WandB 초기화
wandb.init(
    entity=config['wandb']['entity'],
    project=config['wandb']['project'],
    name=config['wandb']['name'],
    config=config
)
print(f"\n✅ WandB 초기화 완료")

# 데이터 로드
print(f"\n📂 전처리된 데이터 로드 중...")
data_dir = os.path.join(config['general']['data_path'], 'processed_data_v5')
train_dataset = load_from_disk(os.path.join(data_dir, 'train'))
eval_dataset = load_from_disk(os.path.join(data_dir, 'eval'))

print(f"  Train: {len(train_dataset)}개")
print(f"  Eval: {len(eval_dataset)}개")

# Tokenizer & Model 로드
print(f"\n🤖 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    config['general']['model_name'],
    cache_dir=config['general'].get('cache_dir', None)
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    config['general']['model_name'],
    cache_dir=config['general'].get('cache_dir', None)
)

print(f"  모델 파라미터 수: {model.num_parameters():,}")

# Data Collator (T5용)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config['general']['output_dir'],
    num_train_epochs=config['training']['num_train_epochs'],
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    
    learning_rate=config['training']['learning_rate'],
    warmup_steps=config['training']['warmup_steps'],
    weight_decay=config['training']['weight_decay'],
    
    save_strategy=config['training']['save_strategy'],
    eval_strategy=config['training']['evaluation_strategy'],
    save_total_limit=config['training']['save_total_limit'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    
    logging_dir=config['training']['logging_dir'],
    logging_steps=config['training']['logging_steps'],
    report_to=config['training']['report_to'],
    
    fp16=config['training']['fp16'],
    gradient_checkpointing=config['training']['gradient_checkpointing'],
    dataloader_num_workers=config['training']['dataloader_num_workers'],
    
    # T5 generation 설정
    predict_with_generate=config['training']['predict_with_generate'],
    generation_max_length=config['training']['generation_max_length'],
    generation_num_beams=config['training']['generation_num_beams'],
)

# Trainer 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print(f"\n🚀 학습 시작!")
print(f"  총 에폭: {config['training']['num_train_epochs']}")
print(f"  실질적 배치 사이즈: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
print(f"  총 스텝: {len(train_dataset) // (config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']) * config['training']['num_train_epochs']}")

# 학습 실행
trainer.train()

# 최종 모델 저장
final_model_path = os.path.join(config['general']['output_dir'], "final_model")
print(f"\n💾 최종 모델 저장 중...")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"\n✅ 학습 완료!")
print(f"  저장 위치: {final_model_path}")

wandb.finish()
