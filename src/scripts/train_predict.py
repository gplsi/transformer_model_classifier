import argparse
import os
import json

from sklearn.model_selection import train_test_split
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict

from src.common.loadData import load_all_data
from src.common.score import scorePredict
from src.model.classification_model import ClassificationModel
from src.common.utils import set_seed


load_dotenv(".env")

def main(parser):
    args = parser.parse_args()
    model_dir = args.model_dir
    label_to_exclude = args.label_to_exclude
    best_result_config = None
    config_path = args.config_path
    model_arg = args.model_arg

    with open(os.getcwd() + config_path) as f:
        general_args = json.load(f)
    df_model_args = pd.read_json(os.getcwd() + model_arg)
    training_args = df_model_args.to_dict(orient='records')[0]
    model_name = general_args["model_name"]

    set_seed(training_args["seed"])

    if model_dir != "":
        model_name = os.getcwd() + model_dir

    df_test = load_all_data(general_args["test_file"], label_to_exclude, general_args["label"],
                                            general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                            name_text_columns =general_args["name_text_columns"])
    #
    labels = list(df_test['labels'].unique())
    if training_args["do_train"]:
        df_train=  load_all_data(general_args["train_file"], label_to_exclude, general_args["label"],
                                                  general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                                name_text_columns =general_args["name_text_columns"])
        labels = list(df_train['labels'].unique())

        df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)

        # Crear un DatasetDict
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_eval),
            "test": Dataset.from_pandas(df_test)
        })
        model = ClassificationModel(model_name, training_args, dataset_dict, general_args)
        model.train()
    else:
        dataset_dict = DatasetDict({
            "test": Dataset.from_pandas(df_test)
        })
        training_args["do_train"] = False
        training_args["do_eval"] = False
        training_args["eval_strategy"]="no"
        model = ClassificationModel(model_name, training_args, dataset_dict, general_args)
    preds = model.predict()
    df_pred = pd.DataFrame(preds, columns=['labels'])
    labels_test = pd.Series(df_test['labels']).to_numpy()
    labels = list(df_test['labels'].unique())
    labels.sort()
    result, f1 = scorePredict(labels_test, df_pred.values, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config_path",
                        default="/config/constructividad/comment.json",
                        type=str,
                        help="File path to configuration parameters.")
    
    parser.add_argument("--model_arg",
                        default="/config/constructividad/comment_model.json",
                        type=str,
                        help="File path to model configuration parameters.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[""],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")

    main(parser)
