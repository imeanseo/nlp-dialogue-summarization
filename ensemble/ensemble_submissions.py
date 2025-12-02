import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from datetime import datetime
import re
import sys


def clean_summary(summary):
    """따옴표, 콤마, 특수문자 완전 제거"""
    if pd.isna(summary):
        return ""
    
    summary = str(summary).strip()
    
    # 따옴표 제거
    if summary.startswith('"') and summary.endswith('"'):
        summary = summary[1:-1]
    summary = summary.replace('""', '')
    
    # 특수문자 제거
    summary = summary.replace('\n', ' ')
    summary = summary.replace('\r', ' ')
    summary = summary.replace('\t', ' ')
    summary = summary.replace(',', '')
    
    # 연속 공백 정리
    summary = re.sub(r'\s+', ' ', summary)
    
    return summary.strip()


def rouge_score_compare(summary1, summary2, summary3):
    """3개 요약 중 서로 가장 유사한 것 선택"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    s12 = scorer.score(summary1, summary2)['rougeL'].fmeasure
    s13 = scorer.score(summary1, summary3)['rougeL'].fmeasure
    s23 = scorer.score(summary2, summary3)['rougeL'].fmeasure
    
    avg1 = (s12 + s13) / 2
    avg2 = (s12 + s23) / 2
    avg3 = (s13 + s23) / 2
    
    scores = [avg1, avg2, avg3]
    best_idx = np.argmax(scores)
    
    return best_idx, [summary1, summary2, summary3][best_idx]


def weighted_rouge_compare(summary1, summary2, summary3, weights=[1.0, 1.0, 1.0]):
    """가중치 기반 ROUGE 선택"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    s12 = scorer.score(summary1, summary2)['rougeL'].fmeasure
    s13 = scorer.score(summary1, summary3)['rougeL'].fmeasure
    s23 = scorer.score(summary2, summary3)['rougeL'].fmeasure
    
    # 가중치 반영
    avg1 = (s12 + s13) / 2 * weights[0]
    avg2 = (s12 + s23) / 2 * weights[1]
    avg3 = (s13 + s23) / 2 * weights[2]
    
    scores = [avg1, avg2, avg3]
    best_idx = np.argmax(scores)
    
    return best_idx, [summary1, summary2, summary3][best_idx]


def length_based_select(summary1, summary2, summary3):
    """중간 길이 선택"""
    lengths = [len(summary1), len(summary2), len(summary3)]
    summaries = [summary1, summary2, summary3]
    sorted_indices = np.argsort(lengths)
    return sorted_indices[1], summaries[sorted_indices[1]]


def post_ensemble_fix(summary):
    """앙상블 후 문법 수정 (개선판)"""
    
    if not summary or pd.isna(summary):
        return summary
    
    # 1. 이중 #Person 태그 제거 ("#Person1#과 #Person2#가 스티븐은" → "#Person1#과 스티븐은")
    summary = re.sub(r'(#Person\d+#[과와])\s+#Person\d+#([은는이가])\s+(\w+)([은는이가])', r'\1 \3\4', summary)
    
    # 2. 중복 조사 제거 ("주디는는" → "주디는")
    summary = re.sub(r'([은는이가을를에])(\1+)', r'\1', summary)
    
    # 3. "#Person2# #Person2#는" → "#Person2#는"
    summary = re.sub(r'(#Person\d+#)\s+\1', r'\1', summary)
    
    # 4. 문장 시작이 조사로 시작하면 #Person1# 추가
    if re.match(r'^[은는이가을를에게]\s', summary):
        summary = '#Person1#' + summary
    
    # 5. 연속 공백 정리
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    return summary


def ensemble_submissions(file1, file2, file3, method='rouge'):
    """3개 submission 앙상블"""

    print(f"\n📂 파일 읽기...")
    df1 = pd.read_csv(file1, encoding='utf-8')
    df2 = pd.read_csv(file2, encoding='utf-8')
    df3 = pd.read_csv(file3, encoding='utf-8')
    
    print(f"✅ 파일1: {len(df1)}개")
    print(f"✅ 파일2: {len(df2)}개")
    print(f"✅ 파일3: {len(df3)}개")
    
    # 정리
    print(f"\n🧹 특수문자 제거 중...")
    df1['summary'] = df1['summary'].apply(clean_summary)
    df2['summary'] = df2['summary'].apply(clean_summary)
    df3['summary'] = df3['summary'].apply(clean_summary)
    print(f"✅ 정리 완료")
    
    if not (len(df1) == len(df2) == len(df3)):
        print("⚠️ 경고: 파일 길이가 다릅니다!")
    
    # 앙상블
    ensemble_summaries = []
    method_counts = {0: 0, 1: 0, 2: 0}
    
    print(f"\n🔀 앙상블 진행 (방법: {method})...")
    
    for i in range(len(df1)):
        s1 = df1['summary'].iloc[i]
        s2 = df2['summary'].iloc[i]
        s3 = df3['summary'].iloc[i]
        
        if method == 'rouge':
            best_idx, best_summary = rouge_score_compare(s1, s2, s3)
        elif method == 'weighted':
            best_idx, best_summary = weighted_rouge_compare(s1, s2, s3, weights=[0.95, 0.93, 1.0])
        elif method == 'length':
            best_idx, best_summary = length_based_select(s1, s2, s3)
        else:
            best_idx, best_summary = 0, s1
        
        # ========================================
        # 후처리 (루프 안에 있어야 함!)
        # ========================================
        best_summary = post_ensemble_fix(best_summary)
        
        ensemble_summaries.append(best_summary)
        method_counts[best_idx] += 1
    
    # 통계
    print(f"\n📊 선택 통계:")
    print(f"  파일1: {method_counts[0]}회 ({method_counts[0]/len(df1)*100:.1f}%)")
    print(f"  파일2: {method_counts[1]}회 ({method_counts[1]/len(df1)*100:.1f}%)")
    print(f"  파일3: {method_counts[2]}회 ({method_counts[2]/len(df1)*100:.1f}%)")
    
    # 결과 저장
    result_df = pd.DataFrame({
        'fname': df1['fname'],
        'summary': ensemble_summaries
    })
    
    date_str = datetime.now().strftime('%m%d_%H%M')
    output_file = f'submission_ensemble_{method}_{date_str}.csv'
    
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 저장 완료: {output_file}")
    
    # 샘플
    print(f"\n🔍 앙상블 결과 (처음 3개):")
    for i in range(min(3, len(result_df))):
        print(f"\n[{result_df['fname'].iloc[i]}]")
        print(f"  {result_df['summary'].iloc[i][:80]}...")
    
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Submission 앙상블 (문법 수정 포함)")
    print("=" * 60)
    
    if len(sys.argv) >= 4:
        file1, file2, file3 = sys.argv[1], sys.argv[2], sys.argv[3]
        method = sys.argv[4] if len(sys.argv) > 4 else 'rouge'
    else:
        print("\n📝 3개 submission 파일:")
        file1 = input("파일1: ").strip()
        file2 = input("파일2: ").strip()
        file3 = input("파일3: ").strip()
        method = input("방법 [rouge/weighted]: ").strip() or 'rouge'
    
    output = ensemble_submissions(file1, file2, file3, method)
    print(f"\n🚀 {output}을 제출하세요!")
