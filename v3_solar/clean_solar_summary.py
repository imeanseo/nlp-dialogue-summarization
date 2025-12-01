import pandas as pd
import re

INPUT_PATH = "./train_solar_fewshot_full.csv"
OUTPUT_PATH = "./train_solar_fewshot_full_cleaned.csv"

def clean_summary(summary: str) -> str:
    """
    solar가 생성한 요약에서 불필요한 메타 텍스트 제거
    """
    # 1) [새 요약], [대체 버전] 같은 태그 제거
    summary = re.sub(r'\[새 요약\]\s*', '', summary)
    summary = re.sub(r'\[대체 버전\]\s*', '', summary)
    
    # 2) ### 로 시작하는 설명 섹션 제거 (### 세부 설명: 등)
    summary = re.sub(r'###\s*.*', '', summary)
    
    # 3) 괄호 안 설명 제거 (기존 요약과 사실 관계... 같은 부분)
    summary = re.sub(r'\(.*?\)', '', summary)
    
    # 4) ※ 참고: 로 시작하는 추가 설명 제거
    summary = re.sub(r'※\s*참고:.*', '', summary, flags=re.DOTALL)
    
    # 5) 연속된 공백/줄바꿈 정리
    summary = re.sub(r'\n+', ' ', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    return summary

def main():
    print(f"📂 로드: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    print(f"🧹 summary 컬럼 정리 중...")
    df['summary'] = df['summary'].apply(clean_summary)
    
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 정리 완료: {OUTPUT_PATH}")
    print(f"   총 {len(df)}개 행")

if __name__ == "__main__":
    main()
