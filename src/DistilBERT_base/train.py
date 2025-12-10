# ============================ Orchestrate Training Process ============================ #
# manage the overall training process, including data loading, model training, validation, and saving the best model.

from src.DistilBERT_base import config
from src.DistilBERT_base import dataset
import torch
import torch.nn as nn
import pandas as pd
from src.DistilBERT_base import engine
import numpy as np
from sklearn import metrics

from sklearn import model_selection
from src.DistilBERT_base.model import DistilBERTBaseUncased
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    print("--- DOWNSAMPLING DATA TO 10,000 SAMPLES FOR CPU TRAINING ---")
    dfx = dfx.sample(n=1000, random_state=config.RANDOM_SEED).reset_index(drop=True)

    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        reviews=df_train.review.values,
        targets=df_train.sentiment.values
    )
    
    # Quick sanity check
    print(f"[DEBUG] train_dataset length: {len(train_dataset)}")
    try:
        sample = train_dataset[0]
        print(f"[DEBUG] Sample keys: {sample.keys()}")
        print(f"[DEBUG] Sample shapes: ids={sample['ids'].shape}, mask={sample['mask'].shape}, targets={sample['targets'].shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load sample from dataset: {e}")
        raise

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    valid_dataset = dataset.BERTDataset(
        reviews=df_valid.review.values,
        targets=df_valid.sentiment.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        # group 1. apply weiht decay (regularization) to all parameters except for the ones in no_decay to prevent overfitting
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        # group 2. no weight decay for parameters in no_decay
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)
    # move model to device (GPU/CPU) before training
    
    
    loss_fn = nn.BCEWithLogitsLoss()

    # training loop
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        print(f"EPOCH {epoch + 1}/{config.EPOCHS}")
        engine.train_fn(
            train_data_loader,
            model,
            optimizer,
            device,
            config.ACCUMULATION_STEPS,
            scheduler,
            loss_fn
        )

        outputs, targets = engine.eval_fn(
            valid_data_loader,
            model,
            device
        )

        outputs = np.array(outputs) >= 0.5 # more than 0.5 as positive sentiment
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy = {accuracy}")
        # save the best model
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()

