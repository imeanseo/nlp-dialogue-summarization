# scripts/v6_train_optuna.py

import os
import yaml
import wandb
import optuna
from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# 경로는 /root/nlp_data 기준에서 실행한다고 가정
TRAIN_PATH = "v6_solar_t5/data/processed/train"
DEV_PATH = "v6_solar_t5/data/processed/dev"
MODEL_NAME = "t5-large"


def load_datasets():
    train_dataset = load_from_disk(TRAIN_PATH)
    eval_dataset = load_from_disk(DEV_PATH)
    return train_dataset, eval_dataset


def load_configs():
    with open("v6_solar_t5/configs/v6_config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open("v6_solar_t5/configs/optuna_config.yaml", "r") as f:
        optuna_cfg = yaml.safe_load(f)
    return base_cfg, optuna_cfg


def build_compute_metrics(tokenizer):
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

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


def model_init():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.gradient_checkpointing_enable()
    return model


def suggest_from_config(trial, space_cfg):
    params = {}

    # learning_rate
    lr_cfg = space_cfg["learning_rate"]
    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        float(lr_cfg["low"]),
        float(lr_cfg["high"]),
        log=bool(lr_cfg.get("log", False)),
    )

    # per_device_train_batch_size
    bs_cfg = space_cfg["per_device_train_batch_size"]
    params["per_device_train_batch_size"] = trial.suggest_categorical(
        "per_device_train_batch_size",
        bs_cfg["values"],
    )

    # num_train_epochs
    ep_cfg = space_cfg["num_train_epochs"]
    params["num_train_epochs"] = trial.suggest_int(
        "num_train_epochs",
        int(ep_cfg["low"]),
        int(ep_cfg["high"]),
    )

    # warmup_steps
    ws_cfg = space_cfg["warmup_steps"]
    params["warmup_steps"] = trial.suggest_int(
        "warmup_steps",
        int(ws_cfg["low"]),
        int(ws_cfg["high"]),
    )
    
      # weight_decay
    wd_cfg = space_cfg.get("weight_decay")
    if wd_cfg is not None:
        params["weight_decay"] = trial.suggest_float(
            "weight_decay",
            float(wd_cfg["low"]),
            float(wd_cfg["high"]),
        )

    return params


def main():
    base_cfg, optuna_cfg = load_configs()
    train_dataset, eval_dataset = load_datasets()

    # W&B 설정 분리
    wandb_cfg = base_cfg.pop("wandb", {})
    run_name = base_cfg.get("run_name", "v6-t5-optuna")

    wandb.login()
    wandb.init(
        project=wandb_cfg.get("project", "dialogue-summarization"),
        entity=wandb_cfg.get("entity", None),
        name=run_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    compute_metrics = build_compute_metrics(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    search_space = optuna_cfg["search_space"]
    n_trials = optuna_cfg.get("n_trials", 2)  # 안전하게 2트라이얼만 먼저

    def objective(trial):
        # Trial별 config 구성
        trial_cfg = base_cfg.copy()
        suggested = suggest_from_config(trial, search_space)
        trial_cfg.update(suggested)

        # Trial별 output 디렉토리 분리
        trial_cfg["output_dir"] = os.path.join(
            base_cfg["output_dir"], f"trial_{trial.number}"
        )

        # 메모리/안정성용 추가 기본값
        trial_cfg.setdefault("per_device_eval_batch_size", 2)
        trial_cfg.setdefault("gradient_accumulation_steps", 8)
        trial_cfg.setdefault("fp16", True)
        trial_cfg.setdefault("save_total_limit", 1)
        trial_cfg.setdefault("seed", 42)
        trial_cfg.setdefault("max_grad_norm", 1.0)

        # TrainingArguments가 모르는 키 제거
        for bad_key in [
            "report_to",
            "run_name",
            "wandb",
        ]:
            trial_cfg.pop(bad_key, None)

        # 디버그용 출력: 실제 들어가는 설정 확인
        print(f"\n[Trial {trial.number}] TrainingArgs config:")
        for k, v in trial_cfg.items():
            print(f"  {k}: {v}")

        training_args = TrainingArguments(
            **trial_cfg,
            report_to=["wandb"],
            run_name=f"{run_name}-trial-{trial.number}",
        )

        print("Train batch example keys:", train_dataset[0].keys())
        
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        wandb.log({f"trial_{trial.number}/eval_rouge": metrics["eval_rouge"]})
        return metrics["eval_rouge"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    best = study.best_trial
    print(f"  Value: {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    wandb.finish()


if __name__ == "__main__":
    main()
