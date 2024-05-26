from tqdm import tqdm
import torch
import numpy as np
import os
import wandb
from sklearn.metrics import accuracy_score


def train(
        task_name,
        train_iter,
        criterion,
        optimizer,
        scheduler,
        model,
        device = 'cuda'
        ):
    
    model = model.to(device)
    model.train()

    batch_loss, preds, trues = list(), list(), list()

    for batch in train_iter:
        inputs, attention_masks, labels = batch['input_ids'], batch['attention_masks'], batch['labels']

        inputs = inputs.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device).long()

        output = model.forward(inputs, attention_masks)
        

        loss = criterion(output.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())
        trues.extend(labels.int().tolist())
        preds.extend(torch.argmax(output.logits, dim=1).tolist())

    scheduler.step()

    train_loss = np.mean(batch_loss)
    train_acc = accuracy_score(trues, preds)

    wandb.log({f"{task_name} train loss" : train_loss,
               f"{task_name} train acc" : train_acc})

def valid(
        task_name,
        valid_iter,
        criterion,
        model,
        save_dir,
        device = 'cuda',
        compare_valid_loss = float('inf')
        ):
    
    model = model.to(device)
    model.eval()

    batch_loss, preds, trues = list(), list(), list()

    with torch.no_grad():
        batch_loss = list()
        for batch in valid_iter:
                
            inputs, attention_masks, labels = batch['input_ids'], batch['attention_masks'], batch['labels']

            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            output = model.forward(inputs, attention_masks)

            loss = criterion(output.logits, labels)

            batch_loss.append(loss.item())
            preds.extend(torch.argmax(output.logits, dim=1).tolist())
            trues.extend(labels.int().tolist())

        valid_loss = np.mean(batch_loss)
        valid_acc = accuracy_score(trues, preds)

        wandb.log({f"{task_name} valid loss" : valid_loss,
                   f"{task_name} valid acc" : valid_acc})

        if compare_valid_loss > valid_loss:

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.module.save_pretrained(f"{save_dir}valid_loss{valid_loss}_model.pt")
            compare_valid_loss = valid_loss

def test(
        test_iter,
        model,
        device = 'cpu'
        ):
    model = model.to(device)
    model.eval()

    preds_list = list()
    inputs_list = list()
    
    with torch.no_grad():
        for batch in tqdm(test_iter, desc = 'predicting ...'):

            inputs, attention_masks = batch['input_ids'], batch['attention_masks']

            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)

            output = model.forward(inputs, attention_masks)

            inputs_list.extend(inputs)
            preds_list.extend(torch.argmax(output.logits, dim=1).tolist())


    return inputs_list, preds_list
