import pandas as pd
import re
import sys

def fix_person_tags(summary):
    """누락된 #Person 태그 복원"""
    
    # 이미 태그 있으면 건드리지 않음
    if '#Person1#' in summary and '#Person2#' in summary:
        return summary
    
    original = summary
    
    # 1. 문장 시작이 조사로 시작하면 #Person1# 추가
    if summary and summary[0] in '은는이가을를':
        if not summary.startswith('#Person'):
            summary = '#Person1#' + summary
    
    # 2. "과 는" → "#Person1#과 #Person2#는"
    summary = re.sub(r'^과\s+는', '#Person1#과 #Person2#는', summary)
    summary = re.sub(r'^와\s+는', '#Person1#과 #Person2#는', summary)
    
    # 3. " 가 " → " #Person1#가 " (첫 번째만)
    if '#Person1#' not in summary:
        summary = summary.replace(' 가 ', ' #Person1#가 ', 1)
    if '#Person1#' not in summary:
        summary = summary.replace(' 는 ', ' #Person1#는 ', 1)
    
    # 4. " 에게" → " #Person2#에게"
    if '#Person2#' not in summary:
        summary = summary.replace(' 에게 ', ' #Person2#에게 ', 1)
    
    # 5. " 의 " 앞에 #Person2# (첫 번째만)
    if '#Person2#' not in summary and ' 의 ' in summary:
        summary = summary.replace(' 의 ', ' #Person2#의 ', 1)
    
    # 6. 여전히 태그 없으면 맨 앞에 추가
    if '#Person1#' not in summary and '#Person2#' not in summary:
        # 영문 이름 있는지 확인
        has_name = bool(re.search(r'\b[A-Z][a-z]+\b', summary))
        if has_name:
            summary = '#Person1#은 ' + summary
        else:
            summary = '#Person1#과 #Person2#가 ' + summary
    
    # 7. #Person1#만 있고 #Person2# 없으면
    elif '#Person1#' in summary and '#Person2#' not in summary:
        # "과", "와" 뒤에 추가
        if ' 과 ' in summary or ' 와 ' in summary:
            summary = re.sub(r'(과|와)\s+', r'\1 #Person2#', summary, count=1)
        elif ' 에게' in summary:
            summary = summary.replace(' 에게', ' #Person2#에게', 1)
    
    # 8. #Person2#만 있고 #Person1# 없으면 (드물지만)
    elif '#Person2#' in summary and '#Person1#' not in summary:
        if not summary.startswith('#Person'):
            summary = '#Person1#은 ' + summary
    
    return summary


def main():
    """submission 파일 수정"""
    
    # 입력 파일명 (명령줄 인자 또는 기본값)
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("수정할 submission 파일명 입력: ").strip()
    
    print(f"\n📂 파일 읽기: {input_file}")
    
    # CSV 읽기
    df = pd.read_csv(input_file)
    
    print(f"✅ 총 {len(df)}개 행 로드")
    
    # 수정 전 샘플
    print("\n🔍 수정 전 샘플 (처음 5개):")
    for i in range(min(5, len(df))):
        print(f"  [{df['fname'].iloc[i]}] {df['summary'].iloc[i][:80]}...")
    
    # Person 태그 수정
    print("\n🔧 #Person 태그 복원 중...")
    df['summary'] = df['summary'].apply(fix_person_tags)
    
    # 수정 후 샘플
    print("\n✅ 수정 후 샘플 (처음 5개):")
    for i in range(min(5, len(df))):
        print(f"  [{df['fname'].iloc[i]}] {df['summary'].iloc[i][:80]}...")
    
    # 통계
    person1_count = df['summary'].str.contains('#Person1#').sum()
    person2_count = df['summary'].str.contains('#Person2#').sum()
    both_count = df['summary'].apply(lambda x: '#Person1#' in x and '#Person2#' in x).sum()
    
    print(f"\n📊 통계:")
    print(f"  #Person1# 포함: {person1_count}/{len(df)}개")
    print(f"  #Person2# 포함: {person2_count}/{len(df)}개")
    print(f"  둘 다 포함: {both_count}/{len(df)}개")
    
    # 저장
    output_file = input_file.replace('.csv', '_fixed.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 저장 완료: {output_file}")
    print(f"\n🚀 이제 {output_file}을 대회에 제출하세요!")


if __name__ == "__main__":
    main()
