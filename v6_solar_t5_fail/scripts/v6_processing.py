# scripts/v6_processing.py

import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "t5-large"


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=150):
    inputs = examples["dialogue"]
    targets = examples["summary"]

    # 입력 토크나이즈
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
    )

    # 출력 토크나이즈
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
        )

    # T5용 labels 세팅
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # CSV 경로
    train_path = "v6_solar_t5/data/augmented/train_v6_perfect.csv"
    dev_path = "v6_solar_t5/data/original/dev.csv"

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    # 토크나이즈
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    dev_tokenized = dev_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names,
    )

    save_dir = "v6_solar_t5/data/processed"
    os.makedirs(save_dir, exist_ok=True)
    train_tokenized.save_to_disk(os.path.join(save_dir, "train"))
    dev_tokenized.save_to_disk(os.path.join(save_dir, "dev"))

    print("전처리 완료 - train/dev 저장 완료")


if __name__ == "__main__":
    main()
