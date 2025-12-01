import pandas as pd
import re
from collections import Counter

def find_all_typos_and_emotions(input_file='train.csv'):
    """
    train.csv에서 모든 오타와 감정 표현을 찾아 분석하는 함수
    """
    print(f"'{input_file}' 파일을 읽고 있습니다...")
    df = pd.read_csv(input_file)
    
    print(f"총 {len(df)}개의 데이터를 분석합니다.\n")
    
    # 패턴별 카운터
    patterns_found = {
        '중간 자음 오타 (ㄱ-ㅎ)': [],
        '중간 모음 오타 (ㅏ-ㅣ)': [],
        '연속된 ㅋ': [],
        '연속된 ㅎ': [],
        '연속된 ㅠ/ㅜ': [],
        '연속된 느낌표 (!!+)': [],
        '연속된 말줄임표 (...+)': [],
        '기타 반복 자음': []
    }
    
    # 정규표현식 패턴들
    choseong = r'[ㄱ-ㅎ]'
    jungseong = r'[ㅏ-ㅣ]'
    
    for idx, row in df.iterrows():
        dialogue = str(row['dialogue'])
        fname = row['fname']
        
        # 1. 중간 자음 오타 찾기 (완성형 한글 + 자음 + 완성형 한글)
        typo_choseong = re.findall(r'[가-힣]' + choseong + r'[가-힣]', dialogue)
        if typo_choseong:
            for match in typo_choseong:
                patterns_found['중간 자음 오타 (ㄱ-ㅎ)'].append({
                    'fname': fname,
                    'pattern': match,
                    'context': dialogue[max(0, dialogue.find(match)-20):dialogue.find(match)+len(match)+20]
                })
        
        # 2. 중간 모음 오타 찾기
        typo_jungseong = re.findall(r'[가-힣]' + jungseong + r'[가-힣\s]', dialogue)
        if typo_jungseong:
            for match in typo_jungseong:
                patterns_found['중간 모음 오타 (ㅏ-ㅣ)'].append({
                    'fname': fname,
                    'pattern': match,
                    'context': dialogue[max(0, dialogue.find(match)-20):dialogue.find(match)+len(match)+20]
                })
        
        # 3. 연속된 ㅋ (2개 이상)
        if re.search(r'ㅋ{2,}', dialogue):
            matches = re.finditer(r'ㅋ{2,}', dialogue)
            for match in matches:
                patterns_found['연속된 ㅋ'].append({
                    'fname': fname,
                    'pattern': match.group(),
                    'context': dialogue[max(0, match.start()-20):min(len(dialogue), match.end()+20)]
                })
        
        # 4. 연속된 ㅎ
        if re.search(r'ㅎ{2,}', dialogue):
            matches = re.finditer(r'ㅎ{2,}', dialogue)
            for match in matches:
                patterns_found['연속된 ㅎ'].append({
                    'fname': fname,
                    'pattern': match.group(),
                    'context': dialogue[max(0, match.start()-20):min(len(dialogue), match.end()+20)]
                })
        
        # 5. 연속된 ㅠ 또는 ㅜ
        if re.search(r'[ㅠㅜ]{2,}', dialogue):
            matches = re.finditer(r'[ㅠㅜ]{2,}', dialogue)
            for match in matches:
                patterns_found['연속된 ㅠ/ㅜ'].append({
                    'fname': fname,
                    'pattern': match.group(),
                    'context': dialogue[max(0, match.start()-20):min(len(dialogue), match.end()+20)]
                })
        
        # 6. 연속된 느낌표 (3개 이상)
        if re.search(r'!{3,}', dialogue):
            matches = re.finditer(r'!{3,}', dialogue)
            for match in matches:
                patterns_found['연속된 느낌표 (!!+)'].append({
                    'fname': fname,
                    'pattern': match.group(),
                    'context': dialogue[max(0, match.start()-20):min(len(dialogue), match.end()+20)]
                })
        
        # 7. 연속된 말줄임표
        if re.search(r'\.{4,}', dialogue):
            matches = re.finditer(r'\.{4,}', dialogue)
            for match in matches:
                patterns_found['연속된 말줄임표 (...+)'].append({
                    'fname': fname,
                    'pattern': match.group(),
                    'context': dialogue[max(0, match.start()-20):min(len(dialogue), match.end()+20)]
                })
        
        # 8. 기타 반복 자음 (ㄴㄴ, ㄷㄷ 등)
        other_repeat = re.findall(r'([ㄴㄷㄹㅁㅂㅅㅈㅊㅌㅍ])\1+', dialogue)
        if other_repeat:
            for match in other_repeat:
                patterns_found['기타 반복 자음'].append({
                    'fname': fname,
                    'pattern': match * 2,  # 최소 2개 반복
                    'context': dialogue
                })
    
    # 결과 출력
    print("="*80)
    print("발견된 오타 및 감정 표현 패턴 분석 결과")
    print("="*80)
    
    for pattern_name, occurrences in patterns_found.items():
        print(f"\n### {pattern_name}")
        print(f"총 {len(occurrences)}건 발견\n")
        
        if len(occurrences) > 0:
            # 상위 5개만 출력
            print(f"[상위 {min(5, len(occurrences))}개 예시]")
            for i, item in enumerate(occurrences[:5], 1):
                print(f"{i}. fname: {item['fname']}")
                print(f"   패턴: '{item['pattern']}'")
                print(f"   문맥: ...{item['context']}...")
                print()
        
        # 패턴 빈도수 통계
        if len(occurrences) > 0:
            pattern_counter = Counter([item['pattern'] for item in occurrences])
            print(f"[빈도수 상위 5개]")
            for pattern, count in pattern_counter.most_common(5):
                print(f"  '{pattern}': {count}회")
            print()
    
    return patterns_found


def create_enhanced_cleaning_function(patterns_found):
    """
    발견된 패턴을 기반으로 향상된 클리닝 함수 생성
    """
    # 실제 데이터에서 발견된 패턴을 반영하여 클리닝 함수를 개선
    print("\n" + "="*80)
    print("권장 수정 사항")
    print("="*80)
    
    print("""
기본 clean_korean_typos 함수 외에 다음 패턴들도 처리하는 것을 권장합니다:

1. 연속된 느낌표/물음표 정규화
   - "!!!" -> "!" (감정은 유지하되 1개로 통일)
   
2. 연속된 말줄임표 정규화
   - "...." -> "..." (3개로 통일)
   
3. 특정 맥락의 자음 처리
   - 실제 발견된 패턴을 기반으로 규칙 추가
    """)


# 실행
if __name__ == "__main__":
    patterns = find_all_typos_and_emotions('train.csv')
    create_enhanced_cleaning_function(patterns)


import pandas as pd
import re

def clean_korean_typos_simple(text):
    """
    실제 발견된 패턴만 처리하는 간단한 클리닝 함수
    
    Args:
        text: 수정할 텍스트
    
    Returns:
        수정된 텍스트
    """
    if pd.isna(text):
        return text
    
    # 1. 연속된 자음을 감정어로 대체
    # ㅎㅎ -> 웃기다
    text = re.sub(r'ㅎ{2,}', ' 웃기다 ', text)
    
    # ㅋㅋ -> 웃기다 (현재는 없지만, 흔한 패턴이므로 포함)
    text = re.sub(r'ㅋ{2,}', ' 웃기다 ', text)
    
    # ㅠㅠ, ㅜㅜ -> 슬프다
    text = re.sub(r'[ㅠㅜ]{2,}', ' 슬프다 ', text)
    
    # 2. 연속된 기호 정규화
    # !!! -> ! (1개로 통일)
    text = re.sub(r'!{2,}', '!', text)
    
    # .... -> ... (3개로 통일)
    text = re.sub(r'\.{4,}', '...', text)
    
    # 3. 연속된 공백 제거
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    
    return text


def process_train_data_simple(input_file='train.csv', output_file='train_cleaned.csv'):
    """
    train.csv를 읽어서 수정하고 저장하는 함수
    """
    print(f"'{input_file}' 파일을 읽고 있습니다...")
    df = pd.read_csv(input_file)
    
    print(f"총 {len(df)}개의 데이터를 처리합니다.\n")
    
    # 수정 전 예시 출력 (발견된 패턴이 있는 데이터)
    print("="*80)
    print("수정 전 샘플")
    print("="*80)
    
    # train_5429 (ㅎㅎ)
    sample_5429 = df[df['fname'] == 'train_5429']
    if not sample_5429.empty:
        print(f"\n1. fname: train_5429 (ㅎㅎ 포함)")
        print(f"   원본: {sample_5429.iloc[0]['dialogue'][:150]}...")
    
    # train_7983 (ㅠㅠ)
    sample_7983 = df[df['fname'] == 'train_7983']
    if not sample_7983.empty:
        print(f"\n2. fname: train_7983 (ㅠㅠ 포함)")
        print(f"   원본: {sample_7983.iloc[0]['dialogue'][:150]}...")
    
    # train_1379 (!!!)
    sample_1379 = df[df['fname'] == 'train_1379']
    if not sample_1379.empty:
        print(f"\n3. fname: train_1379 (!!! 포함)")
        print(f"   원본: {sample_1379.iloc[0]['dialogue'][:150]}...")
    
    # dialogue 컬럼 수정
    print("\n\n오타를 수정하고 있습니다...")
    df['dialogue'] = df['dialogue'].apply(clean_korean_typos_simple)
    
    # summary 컬럼도 있다면 수정
    if 'summary' in df.columns:
        df['summary'] = df['summary'].apply(clean_korean_typos_simple)
        print("summary 컬럼도 함께 수정했습니다.")
    
    # 수정 후 예시 출력
    print("\n" + "="*80)
    print("수정 후 샘플")
    print("="*80)
    
    sample_5429 = df[df['fname'] == 'train_5429']
    if not sample_5429.empty:
        print(f"\n1. fname: train_5429")
        print(f"   수정: {sample_5429.iloc[0]['dialogue'][:150]}...")
    
    sample_7983 = df[df['fname'] == 'train_7983']
    if not sample_7983.empty:
        print(f"\n2. fname: train_7983")
        print(f"   수정: {sample_7983.iloc[0]['dialogue'][:150]}...")
    
    sample_1379 = df[df['fname'] == 'train_1379']
    if not sample_1379.empty:
        print(f"\n3. fname: train_1379")
        print(f"   수정: {sample_1379.iloc[0]['dialogue'][:150]}...")
    
    # 결과 저장
    print(f"\n\n수정된 데이터를 '{output_file}'에 저장하고 있습니다...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("완료!")
    
    # 수정 통계
    print("\n" + "="*80)
    print("수정 통계")
    print("="*80)
    print(f"- 'ㅎㅎ' 처리: 1건 예상")
    print(f"- 'ㅠㅠ' 처리: 1건 예상")
    print(f"- '!!!' 정규화: 5건 예상")
    print(f"- '....' 정규화: 6건 예상")
    
    return df


# 실행
if __name__ == "__main__":
    cleaned_df = process_train_data_simple('train.csv', 'train_cleaned.csv')

df_cleaned = pd.read_csv('train_cleaned.csv')

print("="*80)
print("수정 확인")
print("="*80)

# 1. train_5429 확인 (ㅎㅎ → 웃기다)
sample_5429 = df_cleaned[df_cleaned['fname'] == 'train_5429']
if not sample_5429.empty:
    dialogue = sample_5429.iloc[0]['dialogue']
    print("\n1. train_5429 (ㅎㅎ 수정 확인)")
    print(f"   'ㅎㅎ' 포함 여부: {'ㅎㅎ' in dialogue}")
    print(f"   '웃기다' 포함 여부: {'웃기다' in dialogue}")
    # 해당 부분만 출력
    if '웃기다' in dialogue:
        idx = dialogue.find('웃기다')
        print(f"   문맥: ...{dialogue[max(0, idx-30):idx+50]}...")

# 2. train_7983 확인 (ㅠㅠ → 슬프다)
sample_7983 = df_cleaned[df_cleaned['fname'] == 'train_7983']
if not sample_7983.empty:
    dialogue = sample_7983.iloc[0]['dialogue']
    print("\n2. train_7983 (ㅠㅠ 수정 확인)")
    print(f"   'ㅠㅠ' 포함 여부: {'ㅠㅠ' in dialogue}")
    print(f"   '슬프다' 포함 여부: {'슬프다' in dialogue}")
    # 해당 부분만 출력
    if '슬프다' in dialogue:
        idx = dialogue.find('슬프다')
        print(f"   문맥: ...{dialogue[max(0, idx-30):idx+50]}...")

# 3. train_1379 확인 (!!! → !)
sample_1379 = df_cleaned[df_cleaned['fname'] == 'train_1379']
if not sample_1379.empty:
    dialogue = sample_1379.iloc[0]['dialogue']
    print("\n3. train_1379 (!!! 정규화 확인)")
    print(f"   '!!!' 포함 여부: {'!!!' in dialogue}")
    # '아!' 부분 찾기
    if '아!' in dialogue:
        idx = dialogue.find('아!')
        print(f"   문맥: ...{dialogue[max(0, idx-30):idx+50]}...")

# 4. train_901 확인 (.... → ...)
sample_901 = df_cleaned[df_cleaned['fname'] == 'train_901']
if not sample_901.empty:
    dialogue = sample_901.iloc[0]['dialogue']
    print("\n4. train_901 (.... 정규화 확인)")
    print(f"   '....' 포함 여부: {'....' in dialogue}")
    # '아니에요' 부분 찾기
    if '아니에요' in dialogue:
        idx = dialogue.find('아니에요')
        print(f"   문맥: ...{dialogue[max(0, idx-20):idx+50]}...")

print("\n" + "="*80)
print("전체 통계")
print("="*80)
print(f"총 데이터 수: {len(df_cleaned)}")
print(f"'ㅎㅎ' 남은 개수: {df_cleaned['dialogue'].str.contains('ㅎㅎ', na=False).sum()}")
print(f"'ㅋㅋ' 남은 개수: {df_cleaned['dialogue'].str.contains('ㅋㅋ', na=False).sum()}")
print(f"'ㅠㅠ' 남은 개수: {df_cleaned['dialogue'].str.contains('ㅠㅠ', na=False).sum()}")
print(f"'!!!' 남은 개수: {df_cleaned['dialogue'].str.contains('!!!', na=False, regex=False).sum()}")
print(f"'웃기다' 포함 개수: {df_cleaned['dialogue'].str.contains('웃기다', na=False).sum()}")
print(f"'슬프다' 포함 개수: {df_cleaned['dialogue'].str.contains('슬프다', na=False).sum()}")