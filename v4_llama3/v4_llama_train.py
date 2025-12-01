# ========================================
# v4_train.py
# Llama-3 Korean QLoRA Fine-tuning
# ========================================

import os
import torch
import yaml
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb

logging.set_verbosity_info()

print("=" * 60)
print("v4_train.py - Llama-3 Korean QLoRA Training")
print("=" * 60)

# 1. Config 로드
with open('./v4_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 2. WandB 초기화
wandb.login()
wandb.init(
    entity=config['wandb']['entity'],
    project=config['wandb']['project'],
    name=config['wandb']['name'],
    config=config
)

# 3. 모델 & 토크나이저 설정
model_name = config['general']['model_name']

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

print(f"\n🤖 모델 로드 중: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. LoRA 설정
peft_config = LoraConfig(
    lora_alpha=config['lora']['lora_alpha'],
    lora_dropout=config['lora']['lora_dropout'],
    r=config['lora']['r'],
    bias=config['lora']['bias'],
    task_type=config['lora']['task_type'],
    target_modules=config['lora']['target_modules']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. 데이터셋 로드
print("\n📂 데이터셋 로드 중...")
train_df = pd.read_csv('./processed_data_v4/train.csv')
dev_df = pd.read_csv('./processed_data_v4/dev.csv')

train_dataset = Dataset.from_pandas(train_df[['text']])
eval_dataset = Dataset.from_pandas(dev_df[['text']])

print(f"Train: {len(train_dataset)}")
print(f"Eval:  {len(eval_dataset)}")

# 6. SFTConfig 설정
training_args = SFTConfig(
    output_dir=config['general']['output_dir'],
    num_train_epochs=config['training']['num_train_epochs'],
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    optim=config['training']['optim'],
    save_steps=config['training']['save_steps'],
    logging_steps=10,
    learning_rate=float(config['training']['learning_rate']),
    weight_decay=config['training']['weight_decay'],
    fp16=config['training']['fp16'],
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=config['training']['warmup_ratio'],
    group_by_length=config['training']['group_by_length'],
    lr_scheduler_type=config['training']['lr_scheduler_type'],
    report_to="wandb",
    eval_strategy=config['training']['eval_strategy'],
    eval_steps=config['training']['eval_steps'],
    save_total_limit=config['training']['save_total_limit'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    greater_is_better=config['training']['greater_is_better'],
    max_seq_length=config['tokenizer']['max_length'],
    dataset_text_field="text",
    packing=False,
)

# 7. Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

# 8. 학습 시작
print("\n🚀 학습 시작!")
print(f"📊 학습 설정:")
print(f"  - Model: {model_name}")
print(f"  - Epochs: {config['training']['num_train_epochs']}")
print(f"  - Learning Rate: {config['training']['learning_rate']}")
print(f"  - LoRA r: {config['lora']['r']}")

trainer.train()

# 9. 모델 저장
print("\n💾 모델 저장 중...")
trainer.model.save_pretrained(os.path.join(config['general']['output_dir'], "final_adapter"))
tokenizer.save_pretrained(os.path.join(config['general']['output_dir'], "final_adapter"))

print("✅ 학습 및 저장 완료!")
