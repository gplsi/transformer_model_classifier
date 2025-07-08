import plotly.express as px
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def change_label(row):
    result = 'Incorrect'
    if row['label'] == row['y_pred']:
        result = 'Correct'
    return result

def plot_bar(df):
    df['is_correct'] = df.apply(lambda row: change_label(row), axis=1)
    fig = px.histogram(df, x="label_contradiction", color="is_correct")
    fig.update_layout(xaxis_title="Labels of contradiction", yaxis_title='Examples',
                      legend_title_text="Predictions", margin=dict(l=5, r=10, t=10, b=5),
                      font=dict(size=14))
    fig.show()
    fig.write_image("fig2.svg", width=800, height=300)


def parse_wandb_param(sweep_config, model_args):
    # Extracting the hyperparameter values
    cleaned_args = {}
    layer_params = []
    param_groups = []
    for key, value in sweep_config.items():
        if isinstance(value, dict):
            value = value[0]
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = key.split("_")[-1]
            param_groups.append(
                {
                    "params": [params_key],
                    "lr": value,
                    "weight_decay": model_args.weight_decay
                    if "bias" not in params_key
                    else 0.0,
                }
            )
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups

    # Update the model_args with the extracted hyperparameter values
    model_args.update(cleaned_args)
