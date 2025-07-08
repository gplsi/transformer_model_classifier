from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
import numpy as np




class ClassificationModel:
    def __init__(self, model_id, training_args, dataset_dict, general_args):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
        self.label_encoder = None
        self.training_args = TrainingArguments(**training_args)
        self.general_args = general_args
        self.tokenized_datasets = dataset_dict.map(self.preprocessing_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = None


    def label_encoding(self, examples):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(examples["labels"])
        return self.label_encoder.transform(examples["labels"])

    def preprocessing_function(self, examples):
        if 'text_b' in examples:
            encoded = self.tokenizer(
                examples['text_a'],
                examples['text_b'],
                padding='max_length',
                max_length=self.general_args['max_length'],
                return_tensors="pt",
                truncation=True
            )
        else:
            encoded = self.tokenizer(
                examples['text'],
                padding='max_length',
                max_length=self.general_args['max_length'],
                return_tensors="pt",
                truncation=True
            )

        encoded['label'] = self.label_encoding(examples)
        return encoded

    def train(self):
        self.trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=self.data_collator,
            processing_class=self.tokenizer
        )
        self.trainer.train()

    def predict(self):
        if self.trainer is None:
            self.trainer = Trainer(
                self.model,
                self.training_args,
                data_collator=self.data_collator,
                processing_class=self.tokenizer
            )
        predictions = self.trainer.predict(self.tokenized_datasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        return self.label_encoder.inverse_transform(preds)

