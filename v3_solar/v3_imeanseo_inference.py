# ========================================
# v3_inference.py (v3.5 - 문체 변환 추가)
# ========================================

import torch
import pandas as pd
import yaml
import os
import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import re


def detect_dialogue_style(dialogue: str) -> str:
    """
    대화 문체 감지 (반말 vs 존댓말)
    
    Returns:
        "formal": 존댓말 (요약문 -합니다 체 유지)
        "informal": 반말 (요약문 -한다 체로 변환)
    """
    # 존댓말 패턴
    formal_patterns = [
        r'요[\.\?!,]',        # ~요.
        r'습니다[\.\?!,]',    # ~습니다.
        r'ㅂ니다[\.\?!,]',    # ~ㅂ니다.
        r'세요[\.\?!,]',      # ~세요.
        r'시죠[\.\?!,]',      # ~시죠.
        r'어요[\.\?!,]',      # ~어요.
        r'아요[\.\?!,]',      # ~아요.
        r'해요[\.\?!,]',      # ~해요.
        r'군요[\.\?!,]',      # ~군요.
    ]
    
    # 반말 패턴
    informal_patterns = [
        r'[가-힣]다[\.\?!,]',  # ~다.
        r'[가-힣]야[\.\?!,]',  # ~야.
        r'[가-힣]지[\.\?!,]',  # ~지.
        r'[가-힣][어아][\.\?!,]',  # ~어. ~아.
        r'네[\.\?!,]',         # ~네.
        r'군[\.\?!,]',         # ~군.
    ]
    
    # 카운팅
    formal_count = sum(len(re.findall(p, dialogue)) for p in formal_patterns)
    informal_count = sum(len(re.findall(p, dialogue)) for p in informal_patterns)
    
    # 존댓말이 더 많거나 같으면 formal (기본값)
    return "formal" if formal_count >= informal_count else "informal"


def convert_summary_style(summary: str, style: str) -> str:
    """
    요약문 문체 변환
    
    Args:
        summary: 원본 요약문
        style: "formal" 또는 "informal"
    
    Returns:
        변환된 요약문
    """
    if style == "informal":
        # -합니다/-습니다 → -한다/-다 체 변환
        conversions = [
            (r'합니다\.', '한다.'),
            (r'합니다,', '한다,'),
            (r'습니다\.', '다.'),
            (r'습니다,', '다,'),
            (r'됩니다\.', '된다.'),
            (r'됩니다,', '된다,'),
            (r'입니다\.', '이다.'),
            (r'입니다,', '이다,'),
            (r'있습니다\.', '있다.'),
            (r'있습니다,', '있다,'),
            (r'없습니다\.', '없다.'),
            (r'없습니다,', '없다,'),
            (r'갑니다\.', '간다.'),
            (r'옵니다\.', '온다.'),
            (r'봅니다\.', '본다.'),
            (r'만납니다\.', '만난다.'),
            (r'받습니다\.', '받는다.'),
            (r'줍니다\.', '준다.'),
        ]
        
        for pattern, replacement in conversions:
            summary = re.sub(pattern, replacement, summary)
    
    # formal은 그대로 유지
    return summary


def postprocess_cleanup(text: str) -> str:
    """
    LLM 생성 텍스트 정제 (v3.1 개선판)
    """
    # 1. 프롬프트 잔재 제거
    remove_patterns = [
        r'###?\s*Response:?.*$',
        r'###?\s*Instruction:?.*$',
        r'###?\s*Input:?.*$',
    ]
    for pattern in remove_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 2. 대화문 유출 제거 (강화판)
    dialogue_pattern = r'(#Person\d+#\s*[:：])'
    match = re.search(dialogue_pattern, text)
    if match:
        text = text[:match.start()]
    
    # 3. 줄바꿈 정리
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # 4. 끊긴 문장 처리
    text = text.strip()
    
    if text and text[-1] not in '.!?。':
        last_period = max(
            text.rfind('.'),
            text.rfind('!'),
            text.rfind('?'),
            text.rfind('。')
        )
        
        if last_period > len(text) * 0.5:
            text = text[:last_period + 1]
        elif text:
            text = text.rstrip() + '.'
    
    # 5. 불완전한 문자 제거
    text = re.sub(r'[^\w\s\.,!?;:()#\-가-힣]', '', text)
    
    # 6. 빈 문자열 체크
    text = text.strip()
    if not text or len(text) < 10:
        return "대화 내용을 요약할 수 없습니다."
    
    return text


print("=" * 60)
print("v3_inference.py - LLM Inference (v3.5 문체 변환)")
print("=" * 60)


# 1. Config & 설정
with open('./v3_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ★★★ Adapter 경로 자동 탐색 ★★★
adapter_path = os.path.join(config['general']['output_dir'], "final_adapter")
# adapter_path = "./checkpoints_v3_improved/checkpoint-1500"
# final_adapter가 없으면 가장 최근 checkpoint 사용
if not os.path.exists(adapter_path):
    checkpoint_dir = config['general']['output_dir']
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        adapter_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"⚠️ final_adapter 없음. 최신 checkpoint 사용: {latest_checkpoint}")

print(f"📂 Adapter Path: {adapter_path}")


# 2. 모델 로드
base_model_name = config['general']['model_name']

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

print(f"🤖 Base Model 로드 중: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("🔗 LoRA Adapter 연결 중...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# 3. 데이터 로드
test_df = pd.read_csv('./processed_data_v4/test.csv')

# ★★★ 테스트 모드 선택 ★★★
TEST_MODE = True  # True: 10개만, False: 전체 500개

if TEST_MODE:
    test_df = test_df.head(10)
    print("\n⚠️ 테스트 모드: 10개만 추론")
else:
    print(f"\n🚀 전체 추론 모드: {len(test_df)}개")

# 프롬프트 준비
prompts = [p.rstrip() + "\n" for p in test_df['prompt'].tolist()]

print(f"✅ Test 데이터: {len(prompts)}개")
print(f"📝 첫 프롬프트 끝 부분:")
print(repr(prompts[0][-80:]))


# 4. 추론 루프
results = []
batch_size = 8

# 4. 추론 루프 중 디코딩 부분
for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i : i + batch_size]
    batch_dialogues = test_df.iloc[i:i+batch_size]['dialogue'].tolist()
    
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True,           # padding 추가됨
        truncation=True, 
        max_length=1024
    ).to("cuda")
    
    if i == 0:
        print(f"\n🔍 첫 배치 정보:")
        print(f"  - 토큰화 shape: {inputs['input_ids'].shape}")
        print(f"  - Attention mask shape: {inputs['attention_mask'].shape}")
        print(f"  - 첫 샘플 실제 길이: {inputs['attention_mask'][0].sum().item()}")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    elapsed = time.time() - start_time
    if i == 0:
        print(f"  - 생성 시간: {elapsed:.1f}초")
        total_time = elapsed * (len(prompts) / batch_size)
        print(f"  - 예상 전체 시간: {total_time/60:.1f}분")
    
    # ★★★ 디코딩 개선 ★★★
    for j, output in enumerate(outputs):
        # padding 제외한 실제 입력 길이
        actual_input_length = inputs['attention_mask'][j].sum().item()
        
        # 생성된 부분만 추출
        generated_ids = output[actual_input_length:]
        
        # 디코딩
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 첫 샘플 디버깅
        if i == 0 and j == 0:
            print(f"\n📝 디코딩 디버깅:")
            print(f"  - 전체 출력 길이: {len(output)} 토큰")
            print(f"  - 실제 입력 길이: {actual_input_length} 토큰")
            print(f"  - 생성된 길이: {len(generated_ids)} 토큰")
            print(f"  - 생성 텍스트 (처음 200자): {generated_text[:200]}")
        
        # 정제
        summary = postprocess_cleanup(generated_text)
        
        # 문체 변환
        dialogue_style = detect_dialogue_style(batch_dialogues[j])
        summary = convert_summary_style(summary, dialogue_style)
        
        # 첫 샘플 최종 결과
        if i == 0 and j == 0:
            print(f"\n🎨 문체 변환:")
            print(f"  - 감지된 문체: {dialogue_style}")
            print(f"  - 최종 요약: {summary}")
        
        results.append(summary)



# 5. 저장
print("\n💾 제출 파일 저장 중...")
from datetime import datetime
import pytz  # 시간대 라이브러리

# ★★★ 한국 시간으로 변경 ★★★
kst = pytz.timezone('Asia/Seoul')
date_str = datetime.now(kst).strftime('%m%d_%H%M') 
mode_suffix = "test10" if TEST_MODE else "full"
filename = f'submission_v4_{mode_suffix}_{date_str}.csv'

submission = pd.DataFrame({
    'fname': [f'test_{i}' for i in range(len(results))],
    'summary': results
})

os.makedirs('./predictions', exist_ok=True)
submission.to_csv(f'./predictions/{filename}', index=False)

print(f"✅ 완료! ./predictions/{filename}")

# 샘플 확인
print("\n📝 생성 샘플:")
for i in range(min(5, len(results))):
    print(f"[{i}] {results[i]}")
    print("-" * 30)
