# transformer_model_classifier

## Run training script
```bash
python -m src.scripts.train_predict --config_path /config/constructividad/comment.json --model_arg /config/constructividad/comment_model.json
```

## Code configuration
```json
    {
        "model_type": "roberta",
        "model_name": "PlanTL-GOB-ES/roberta-base-bne",
        "wandb_project": "comment_const_active_learning",
        "best_result_config": "",
        "train_file": "/data/constructividad/train.tsv",
        "test_file": "/data/constructividad/test.tsv",
        "output_dir": "comment_const_1",
        "label": "CONST",
        "filter_label": "",
        "filter_label_value": "",
        "features_name": [],
        "two_text": false,
        "name_text_columns": ["text"],
        "max_length": 256
    }

```

## Models configurations

Use the same hyperparameters of TrainingArguments. Follow the example below to create your own configuration file.

```json
    {
      "learning_rate": 2e-5,
      "num_train_epochs": 2,
      "per_device_train_batch_size": 16,
      "per_device_eval_batch_size": 16,
      "overwrite_output_dir": true,
      "seed": 852,
      "fp16": true,
      "report_to": "tensorboard",
      "fp16_opt_level": "01",
      "do_train": true,
      "do_eval": true,
      "do_predict": true,
      "save_strategy": "epoch",
      "eval_strategy": "epoch",
      "output_dir": "./results",
      "no_cuda": true
    }
```

