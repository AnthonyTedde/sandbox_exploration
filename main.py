import argparse
from os import walk, path
import mlflow
import collections.abc
import core.usecases

models_list = [
    a.removesuffix(".py")
    for a in next(walk("core/usecases"), (None, None, []))[2]
    if not a.startswith("_")
]


def parse_args():
    parser = argparse.ArgumentParser(description="NLP from Databricks")
    parser.add_argument("--run_training_for", choices=models_list)
    parser.add_argument("--get_model_list", action="store_true")
    # parser.add_argument("--task", type=str, default=None)
    # parser.add_argument("--model_checkpoint", type=str, default=None)
    # parser.add_argument("--batch_size", type=str, default=None)
    # parser.add_argument("--huggingface_token", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with mlflow.start_run():
        print("popo")
        mlflow.log_param("lr", 0.001)
        # Your ml code
        mlflow.log_metric("val_loss", 1)
    # if args.get_model_list:
    #     print(models_list)
    #     return 0


if __name__ == "__main__":
    main()
