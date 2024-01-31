import transformers, torch
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, AutoModelForMaskedLM, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging, tqdm, os, sys, pathlib, argparse, numpy as np, pandas as pd

class TextClassifier(nn.Module):

    def __init__(self, n_classes, transformer):
        super(TextClassifier, self).__init__()
        self.bert = transformer
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)


    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)
        

def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples,
        logger
):
    model = model.train()

    losses = []
    correct_predictions = 0
    for i, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
    return correct_predictions.double() /n_examples, np.mean(losses)



