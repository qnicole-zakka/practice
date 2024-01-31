import transformers, torch
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging, tqdm, os, sys, pathlib, argparse, numpy as np, pandas as pd
from text_data import TextDataset, create_data_loader
from text_util import TextClassifier, train_epoch, eval_model, get_predictions, show_confusion_matrix, show_classification_report


torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# loggers


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
transformer = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)


MAX_LEN = 512
BATCH_SIZE = 8
N_CLASSES = 2

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
model = TextClassifier(N_CLASSES, transformer)
# Ensure all layers are unfrozen for fine-tuning
for param in model.parameters():
    param.requires_grad = True

EPOCHS = 30
LR = 0.0001
N_CLASSES = 2
optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_data_loader) * EPOCHS
)
loss_fn = nn.CrossEntropyLoss().to(device)
model = model.to(device)


for e in range(EPOCHS):
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df),
        logger
    )


    outputs, targets = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(test_df),
        logger
    )
    preds = get_predictions(outputs)
    show_confusion_matrix(targets, preds, logger)
    show_classification_report(targets, preds, logger)
    logger.info(f'Epoch {e + 1}/{EPOCHS}')
    logger.info(f'Test Loss: {np.mean(losses):.3f}')
    logger.info(f'Test Accuracy: {correct_predictions / n_examples * 100:.3f}%')
    logger.info('-' * 10)

    if val_acc > best_acc:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_acc = val_acc