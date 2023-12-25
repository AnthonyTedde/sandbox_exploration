import argparse

import datasets
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, get_dataset_config_info
from evaluate import load, MetricInfo

# TODO
#  [] Consider get_dataset_config_info to get information for glue: mnli -> match and missmatch

TASK = "qnli"
MODEL_CHECKPOINT = "xlm-roberta-base"
BATCH_SIZE = 16

METRICS = {
    "qnli": "accuracy",
}

# TODO:
#  [] Put the followin in configuation file
DEBUG = True


def build_dataset_and_metrics(task=TASK):
    # TODO:
    #  [] Create datasets and incorporate it into a datasetdict
    #  [] Create a configuration file
    # train = load_dataset(path="glue", name=task, split=datasets.ReadInstruction('train', to=20))
    # validation = load_dataset(path="glue", name=task, split=datasets.ReadInstruction("validation", to=20))
    # test = load_dataset(path="glue", name=task, split=datasets.ReadInstruction("test", to=20))
    # dataset_dct = datasets.DatasetDict({'train': train, 'validation': validation, 'test': test})
    return [
        load_dataset(path="glue", name=task),
        # dataset_dct,
        load(path="glue", config_name=task)
    ]


def build_autoclass_from_pretrained(
        auto_cls,
        model_ckpt=MODEL_CHECKPOINT,
        *args, **kwargs
):
    obj = auto_cls.from_pretrained(
        pretrained_model_name_or_path=model_ckpt,
        *args,
        **kwargs
    )
    return obj


def get_features(dataset):
    keys = dataset["train"].features.keys()
    return [i for i in keys if i not in ['label', 'idx', "label_name"]]


def parse_args():
    parser = argparse.ArgumentParser(description="NLP from Databricks")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--huggingface_token", type=str, default=None)
    return parser.parse_args()


def preprocess_text(dataset, features, tokenizer):
    def tokenize(batch):
        dat = [batch[f] for f in features]
        return tokenizer(*dat, truncation=True)

    encoded_data = dataset.map(tokenize, batched=True)
    return encoded_data


def get_label_size(encoded_data):
    label = np.ravel(encoded_data["train"].data["label"])
    return np.unique(label).size


def get_training_argument(model,
                          encoded_data,
                          metrics,
                          batch_size,
                          push_to_hub=None):
    model.num_labels = get_label_size(encoded_data)

    model_name = model.name_or_path
    task = metrics.config_name
    args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-{task}",
        evaluation_strategy='epoch',
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=METRICS[task],
        push_to_hub=push_to_hub,
    )
    return args


def compute_metrics(metrics):
    def compute(eval_pred, metrics=metrics):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metrics.compute(
            predictions=predictions,
            references=labels
        )

    return compute


def finetune(model, args, encoded_data, tokenizer, metrics):
    # --> Change validation for other glue task
    encoded_data
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_data["train"],
        eval_dataset=encoded_data["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(metrics=metrics)
    )
    return trainer


def main():
    args = parse_args()

    task = args.task if args.task else TASK
    model_ckpt = args.model_checkpoint if args.model_checkpoint else MODEL_CHECKPOINT
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    huggingface_token = args.huggingface_token

    dataset, metrics = build_dataset_and_metrics(task=task)
    # change label --> label_name (entailment ...)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = build_autoclass_from_pretrained(
        auto_cls=AutoTokenizer,
        model_ckpt=model_ckpt,
        use_fast=True
    )
    model = build_autoclass_from_pretrained(
        auto_cls=AutoModelForSequenceClassification,
        model_ckpt=model_ckpt,
    ).to(device)
    features = get_features(dataset)

    encoded_data = preprocess_text(dataset, features, tokenizer)
    args = get_training_argument(model, encoded_data, metrics, batch_size, push_to_hub=bool(huggingface_token))

    trainer = finetune(model, args, encoded_data, tokenizer, metrics)

    trainer.train()
    trainer.evaluate()

    if huggingface_token:
        from huggingface_hub import login
        login(token=huggingface_token)
        trainer.push_to_hub(commit_message="dry_run_test")


if __name__ == '__main__':
    main()
