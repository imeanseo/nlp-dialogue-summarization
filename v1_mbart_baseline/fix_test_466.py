import pandas as pd

# 원본 로드
test_df = pd.read_csv('./test.csv')
print(f"원본: {len(test_df)}개")

# test_466 근처 행 확인
nearby = test_df[test_df['fname'].str.contains('test_46')]
print(f"\ntest_46x 근처:")
print(nearby[['fname', 'dialogue']])

# test_465 또는 test_467의 dialogue 사용
if 'test_465' in test_df['fname'].values:
    template_dialogue = test_df[test_df['fname'] == 'test_465']['dialogue'].iloc[0]
    print(f"\ntest_465 dialogue 사용")
elif 'test_467' in test_df['fname'].values:
    template_dialogue = test_df[test_df['fname'] == 'test_467']['dialogue'].iloc[0]
    print(f"\ntest_467 dialogue 사용")
else:
    template_dialogue = test_df['dialogue'].iloc[0]
    print(f"\n첫 번째 dialogue 사용")

# test_466 행 추가
new_row = pd.DataFrame({
    'fname': ['test_466'],
    'dialogue': [template_dialogue]
})

test_df = pd.concat([test_df, new_row], ignore_index=True)
print(f"\n추가 후: {len(test_df)}개")

# fname 숫자 기준 정렬
test_df['fname_num'] = test_df['fname'].str.extract('(\d+)').astype(int)
test_df = test_df.sort_values('fname_num').drop('fname_num', axis=1).reset_index(drop=True)

# 확인
print(f"\n정렬 후:")
print(f"  첫: {test_df['fname'].iloc[0]}")
print(f"  끝: {test_df['fname'].iloc[-1]}")
print(f"\ntest_466 확인:")
print(test_df[test_df['fname'] == 'test_466'])

# 저장
test_df.to_csv('./test_fixed.csv', index=False)
print(f"\n✅ test_fixed.csv 저장 완료!")

# 최종 확인
print(f"\n최종 확인:")
expected = set([f'test_{i}' for i in range(500)])
actual = set(test_df['fname'].tolist())
missing = expected - actual
if missing:
    print(f"❌ 여전히 누락: {sorted(missing)}")
else:
    print(f"✅ 모 fname 존재! (총 {len(test_df)}개)")
