# ========================================
# v2_train.py
# mBART Fine-tuning
# ========================================

import pandas as pd
import numpy as np
import os
import yaml
import torch
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from rouge import Rouge, rouge
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("v1_train.py - mBART Fine-tuning")
print("=" * 60)

# ========================================
# 1. Config 불러오기
# ========================================

print("\n📖 Config 불러오기...")
config_path = './v2_config.yaml'

with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

config['training']['learning_rate'] = float(config['training']['learning_rate'])
config['training']['num_train_epochs'] = int(config['training']['num_train_epochs'])
config['training']['per_device_train_batch_size'] = int(config['training']['per_device_train_batch_size'])
config['training']['per_device_eval_batch_size'] = int(config['training']['per_device_eval_batch_size'])
config['training']['warmup_ratio'] = float(config['training']['warmup_ratio'])
config['training']['weight_decay'] = float(config['training']['weight_decay'])
config['training']['gradient_accumulation_steps'] = int(config['training']['gradient_accumulation_steps'])
config['training']['save_steps'] = int(config['training']['save_steps'])
config['training']['eval_steps'] = int(config['training']['eval_steps'])
config['training']['save_total_limit'] = int(config['training']['save_total_limit'])
config['training']['seed'] = int(config['training']['seed'])
config['training']['logging_steps'] = int(config['training']['logging_steps'])
config['training']['generation_max_length'] = int(config['training']['generation_max_length'])
config['training']['early_stopping_patience'] = int(config['training']['early_stopping_patience'])
config['training']['early_stopping_threshold'] = float(config['training']['early_stopping_threshold'])
config['tokenizer']['encoder_max_len'] = int(config['tokenizer']['encoder_max_len'])
config['tokenizer']['decoder_max_len'] = int(config['tokenizer']['decoder_max_len'])

print("✅ Config 로드 완료")
print(f"  Model: {config['general']['model_name']}")
print(f"  Encoder Max: {config['tokenizer']['encoder_max_len']}")
print(f"  Decoder Max: {config['tokenizer']['decoder_max_len']}")
print(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
print(f"  Learning Rate: {config['training']['learning_rate']}")
print(f"  Epochs: {config['training']['num_train_epochs']}")

# ========================================
# 2. GPU 확인
# ========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Device: {device}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠️ GPU 없음 - CPU로 학습 (매우 느림)")

# ========================================
# 3. WandB 초기화
# ========================================

try:
    import wandb
    
    print("\n🔗 WandB 초기화...")
    
    # WandB 로그인 체크
    try:
        wandb.login()
    except:
        print("⚠️ WandB 로그인 필요: wandb login")
    
    # 초기화
    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config,
        resume='allow',  # 중단 시 재개 가능
    )
    
    # Baseline 스타일 환경변수
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "false"
    
    print(f"✅ WandB 초기화 완료")
    print(f"📊 Dashboard: {wandb.run.get_url()}")
    use_wandb = True
    
except Exception as e:
    print(f"\n⚠️ WandB 초기화 실패: {e}")
    print("WandB 없이 계속 진행합니다.")
    use_wandb = False
    config['training']['report_to'] = 'none'

# ========================================
# 4. 데이터 로드 (전처리된 데이터)
# ========================================

print("\n📂 전처리된 데이터 로딩...")

train_df = pd.read_csv('./processed_data/train_processed.csv')
dev_df = pd.read_csv('./processed_data/dev_processed.csv')

print(f"✅ Train: {len(train_df):,}개")
print(f"✅ Dev:   {len(dev_df):,}개")

# ========================================
# 5. 토크나이저 & 모델 로드
# ========================================

print(f"\n🔤 토크나이저 로드: {config['general']['model_name']}")

tokenizer = AutoTokenizer.from_pretrained(
    config['general']['model_name'],
    src_lang='ko_KR',
    tgt_lang='ko_KR'
)

# Special Token 추가
special_tokens = config['tokenizer']['special_tokens']
num_added = tokenizer.add_special_tokens({
    'additional_special_tokens': special_tokens
})
print(f"✅ {num_added}개 Special Token 추가")

# 체크포인트 확인
import glob
checkpoints = glob.glob(os.path.join(config['general']['output_dir'], "checkpoint-*"))
last_checkpoint = None

if checkpoints:
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    last_checkpoint = checkpoints[-1]
    print(f"\n📥 체크포인트 발견: {last_checkpoint}")

# 모델 로드
print(f"\n🤖 모델 로드...")
if last_checkpoint:
    print(f"체크포인트에서 재개: {last_checkpoint}")
    model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
else:
    print(f"사전학습 모델 로드: {config['general']['model_name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(config['general']['model_name'])

# Vocab 크기 조정
model.resize_token_embeddings(len(tokenizer))
print(f"✅ Vocab 크기: {len(tokenizer)}")

# GPU로 이동
model.to(device)

# ========================================
# 6. Dataset 준비
# ========================================

print("\n📊 Dataset 생성...")

def preprocess_function(examples):
    """토크나이징 함수"""
    
    # 입력 (대화문)
    inputs = examples['dialogue_clean']
    targets = examples['summary_clean']
    
    # 토크나이징
    model_inputs = tokenizer(
        inputs,
        max_length=config['tokenizer']['encoder_max_len'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 타겟 토크나이징
    labels = tokenizer(
        targets,
        max_length=config['tokenizer']['decoder_max_len'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# HuggingFace Dataset으로 변환
train_dataset = Dataset.from_pandas(train_df[['dialogue_clean', 'summary_clean']])
val_dataset = Dataset.from_pandas(dev_df[['dialogue_clean', 'summary_clean']])

# 토크나이징 적용
print("토크나이징 중...")
train_dataset = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['dialogue_clean', 'summary_clean']
)
val_dataset = val_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['dialogue_clean', 'summary_clean']
)

print(f"✅ Train Dataset: {len(train_dataset):,}개")
print(f"✅ Val Dataset: {len(val_dataset):,}개")

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ========================================
# 7. ROUGE 평가 함수 수정
# ========================================

def compute_metrics(eval_pred):
    """ROUGE 점수 계산"""
    predictions, labels = eval_pred
    
    # -100을 패딩으로 처리
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    
    rouge_scorer = Rouge()
    
    def clean_text(text):
        """모델 토큰만 제거, #Person#은 유지"""
        # 1) 모델 특수 토큰 제거
        remove_tokens = ['<usr>', '</s>', '<s>', '<pad>', '<unk>', 
                        'ko_KR', 'en_XX', 'ja_XX', 'zh_CN', '__', '▁']
        for token in remove_tokens:
            text = text.replace(token, '')
        
        # 2) 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 3) 주어 누락 수정 
        # "는 에게" → "#Person1#는 #Person2#에게"
        if text.startswith('는 '):
            text = '#Person1#' + text
        text = re.sub(r'([.!?])\s+는\s', r'\1 #Person1#는 ', text)
        text = re.sub(r'(\w+)는\s+에게', r'\1는 #Person2#에게', text)
        
        return text if text else "empty"
    
    # 후처리 적용
    replaced_preds = [clean_text(p) for p in decoded_preds]
    replaced_labels = [clean_text(l) for l in decoded_labels]
    
    # ROUGE 계산
    try:
        results = rouge_scorer.get_scores(replaced_preds, replaced_labels, avg=True)
        
        # 샘플 출력
        print("\n" + "-" * 60)
        print(f"[예측] {replaced_preds[0]}")
        print(f"[정답] {replaced_labels[0]}")
        print("-" * 60)
        
        return {
            'rouge1': results['rouge-1']['f'],
            'rouge2': results['rouge-2']['f'],
            'rougeL': results['rouge-l']['f'],
        }
    except Exception as e:
        print(f"⚠️ ROUGE 계산 실패: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


# ========================================
# 8. Training Arguments
# ========================================

print("\n⚙️ Training Arguments 설정...")

training_args = Seq2SeqTrainingArguments(
    # 출력 디렉토리
    output_dir=config['general']['output_dir'],
    
    # 학습 (직접 타입 변환!)
    num_train_epochs=int(config['training']['num_train_epochs']),
    per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
    per_device_eval_batch_size=int(config['training']['per_device_eval_batch_size']),
    gradient_accumulation_steps=int(config['training']['gradient_accumulation_steps']),
    
    # 최적화 (직접 타입 변환!)
    learning_rate=float(config['training']['learning_rate']),
    weight_decay=float(config['training']['weight_decay']),
    warmup_ratio=float(config['training']['warmup_ratio']),
    lr_scheduler_type=str(config['training']['lr_scheduler_type']),
    optim=str(config['training']['optim']),
    
    # 평가
    eval_strategy='steps',
    eval_steps=int(config['training']['eval_steps']),
    
    # 체크포인트
    save_strategy='steps',
    save_steps=int(config['training']['save_steps']),
    save_total_limit=int(config['training']['save_total_limit']),
    load_best_model_at_end=bool(config['training']['load_best_model_at_end']),
    metric_for_best_model='rougeL',
    greater_is_better=True,
    
    # 생성
    predict_with_generate=True,
    generation_max_length=int(config['tokenizer']['decoder_max_len']),
    
    # 효율성
    fp16=bool(config['training']['fp16']),
    
    # 로깅
    logging_dir=str(config['training']['logging_dir']),
    logging_strategy='steps',
    logging_steps=int(config['training']['logging_steps']),
    report_to=str(config['training']['report_to']),
    
    # 기타
    seed=int(config['training']['seed']),
    overwrite_output_dir=bool(config['training']['overwrite_output_dir']),
    do_train=bool(config['training']['do_train']),
    do_eval=bool(config['training']['do_eval']),
    ignore_data_skip=False,
)

print("✅ Training Arguments 준비 완료")

# ========================================
# 9. Trainer 생성
# ========================================

print("\n🎯 Trainer 생성...")

# Early Stopping Callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=config['training']['early_stopping_patience'],
    early_stopping_threshold=config['training']['early_stopping_threshold']
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

print("✅ Trainer 준비 완료")

# ========================================
# 10. 학습 시작
# ========================================

print("\n" + "=" * 60)
print("🚀 학습 시작!")
print("=" * 60)
print(f"  Epochs: {config['training']['num_train_epochs']}")
print(f"  Batch Size: {config['training']['per_device_train_batch_size']}")
print(f"  Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
print(f"  Effective Batch: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
print(f"  Learning Rate: {config['training']['learning_rate']}")
print(f"  Save Steps: {config['training']['save_steps']}")
print("=" * 60)

try:
    # 체크포인트 재개
    if last_checkpoint:
        print(f"\n📥 체크포인트에서 재개: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🆕 처음부터 학습 시작")
        trainer.train()
    
    print("\n✅ 학습 완료!")
    
    # Best 모델 저장
    best_model_path = os.path.join(config['general']['output_dir'], 'best_model')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\n💾 Best 모델 저장: {best_model_path}")
    
    # WandB에 모델 업로드
    if use_wandb:
        artifact = wandb.Artifact('mbart-best-model', type='model')
        artifact.add_dir(best_model_path)
        wandb.log_artifact(artifact)
        print("📤 WandB에 모델 업로드 완료")

except KeyboardInterrupt:
    print("\n⚠️ 학습 중단 (Ctrl+C)")
    print("💾 현재 체크포인트가 저장되었습니다.")
    print("다시 실행하면 마지막 체크포인트에서 재개됩니다.")

except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU 메모리 정리 완료")
    
    # WandB 종료
    if use_wandb:
        wandb.finish()

# ========================================
# 완료
# ========================================

print("\n" + "=" * 60)
print("✅ v1_train.py 완료!")
print("=" * 60)
print("\n📁 생성된 파일:")
print(f"  - {config['general']['output_dir']}/best_model/")
print(f"  - {config['general']['output_dir']}/checkpoint-*/")
print(f"  - {config['training']['logging_dir']}/")

if use_wandb and wandb.run is not None:
    print(f"\n📊 WandB Dashboard:")
    print(f"  {wandb.run.get_url()}")

print("\n🚀 다음 단계:")
print("  python v1_inference.py  # 추론 및 제출 파일 생성")
print("=" * 60)
