import numpy as np
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Benchmarker:
    encoded_data: DatasetDict = None
    training_ds_name: str = "train"
    validation_ds_name: str = "validation"
    test_ds_name: str = "test"
    _trainer: Trainer = None

    def __init__(self, model_ckpt, dataset, metric, task=None, training_argument=None):
        self.model_ckpt = model_ckpt
        self.task = task
        self.dataset = dataset
        self.metric = metric
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ckpt=self.model_ckpt, use_fast=True
        )
        self.model = AutoModelForSequenceClassification(
            model_ckpt=AutoModelForSequenceClassification
        )
        self.training_argument = (
            training_argument if training_argument else TrainingArguments()
        )

    def preprocess_data(self):
        def tokenize(batch):
            dat = [batch[f] for f in self.features]

        self.encoded_data = self.dataset.map(tokenize, batched=True)

    def finetuner(self) -> None:
        if not self.encoded_data:
            # TODO
            #  [] Raise error if data not yet encoded.
            pass
        self._trainer = Trainer(
            model=self.model,
            args=self.training_argument,
            train_dataset=self.encoded_data[self.training_ds_name],
            eval_dataset=self.encoded_data[self.validation_ds_name],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

    # -----------
    # Properties
    # -----------

    @property
    def features(self):
        keys = self.datasets["train"].features.keys()
        return [i for i in keys if i not in ["label", "idx"]]

    @property
    def trainer(self):
        return self._trainer

    # -----------
    # Private
    # -----------

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)
