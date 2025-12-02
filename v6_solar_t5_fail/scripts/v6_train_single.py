import os
import yaml
from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import wandb


MODEL_NAME = "t5-large"
TRAIN_PATH = "v6_solar_t5/data/processed/train"
DEV_PATH = "v6_solar_t5/data/processed/dev"


def load_datasets():
    train_dataset = load_from_disk(TRAIN_PATH)
    eval_dataset = load_from_disk(DEV_PATH)
    return train_dataset, eval_dataset


def load_config():
    with open("v6_solar_t5/configs/v6_config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_compute_metrics(tokenizer):
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        try:
            preds = preds.tolist()
        except AttributeError:
            pass

        try:
            labels = labels.tolist()
        except AttributeError:
            pass

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
        result["rouge"] = (
            result["rouge1"] + result["rouge2"] + result["rougeL"]
        ) / 3
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def main():
    cfg = load_config()
    train_dataset, eval_dataset = load_datasets()

    wandb_cfg = cfg.pop("wandb")
    run_name = cfg.get("run_name", "v6-t5-single")

    wandb.login()
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=run_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = build_compute_metrics(tokenizer)

    lr = 5e-5  # 기본 learning_rate 설정

    training_args = TrainingArguments(
        **cfg,
        learning_rate=lr,
        eval_accumulation_steps=8,
    )

    print("Train keys:", train_dataset[0].keys())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final metrics:", metrics)
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
