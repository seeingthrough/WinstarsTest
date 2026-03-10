import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT for Animal NER")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON dataset")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./ner_model", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    return parser.parse_args()


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['ner_tags']

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

        labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                label_str = tags[word_idx]
                labels.append(self.label2id.get(label_str, 0))
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        item_out = {key: torch.as_tensor(val) for key, val in encoding.items() if key != 'offset_mapping'}
        item_out['labels'] = torch.as_tensor(labels)
        return item_out


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_predictions_flat = [item for sublist in true_predictions for item in sublist]
    true_labels_flat = [item for sublist in true_labels for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, true_predictions_flat, average='macro')
    acc = accuracy_score(true_labels_flat, true_predictions_flat)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    args = parse_args()

    with open(args.data_path, 'r') as f:
        raw_data = json.load(f)
    unique_tags = set(tag for doc in raw_data for tag in doc['ner_tags'])
    tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
    id2tag = {id: tag for tag, id in tag2id.items()}

    print(f"Found tags: {tag2id}")

    train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = NERDataset(train_data, tokenizer, tag2id)
    val_dataset = NERDataset(val_data, tokenizer, tag2id)

    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
    )

    # Аргументы тренера
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()