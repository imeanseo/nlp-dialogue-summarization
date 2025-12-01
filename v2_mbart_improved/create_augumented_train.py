import pandas as pd
import os
import re

INPUT_PATH = "./train.csv"          # 원본 학습 데이터
OUTPUT_PATH = "./train_augmented.csv"  # 증강 후 저장 파일

def speaker_swap(text: str) -> str:
    """
    #Person1# ↔ #Person2# 스왑.
    """
    if not isinstance(text, str):
        text = str(text)

    # 임시 토큰을 써서 겹치지 않게 치환
    text = text.replace("#Person1#", "#TMP_PERSON#")
    text = text.replace("#Person2#", "#Person1#")
    text = text.replace("#TMP_PERSON#", "#Person2#")
    return text

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"원본 데이터 개수: {len(df)}")

    augmented_rows = []

    for idx, row in df.iterrows():
        # 원본 그대로
        augmented_rows.append(row.to_dict())

        # 스왑 버전 생성
        swapped_dialogue = speaker_swap(row["dialogue"])
        swapped_summary = speaker_swap(row["summary"])

        new_fname = f"{row['fname']}_swap"

        augmented_rows.append({
            "fname": new_fname,
            "dialogue": swapped_dialogue,
            "summary": swapped_summary,
            "topic": row["topic"],
        })

        if (idx + 1) % 1000 == 0:
            print(f"{idx+1}개 처리 중...")

    aug_df = pd.DataFrame(augmented_rows)
    print(f"증강 후 데이터 개수: {len(aug_df)}")

    aug_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
