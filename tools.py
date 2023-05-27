import os
import tarfile
from glob import glob

import albumentations as A
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from easydict import EasyDict
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from scipy.sparse import csr_matrix
from tqdm import tqdm

shuffle = True
num_workers = 10
batch_size = 128
image_size = 64


def padding(data1, data2, pad_idx=0):
    len1, len2 = data1.shape[1], data2.shape[1]
    if len1 > len2:
        data2 = torch.cat(
            [data2, torch.ones(data2.shape[0], len1 - len2).long() * pad_idx], dim=1
        )
    elif len2 > len1:
        data1 = torch.cat(
            [data1, torch.ones(data1.shape[0], len2 - len1).long() * pad_idx], dim=1
        )
    return data1, data2


def batch_padding(data, pad_idx=0):
    max_len = 0
    for text in data:
        max_len = max(max_len, len(text))
    for i in range(len(data)):
        data[i] += [pad_idx] * (max_len - len(data[i]))
    return torch.tensor(data)


def convertToTorchFloat(x):
    x = torch.from_numpy(x).float()
    return x


def convertToTorchInt(x):
    x = torch.from_numpy(x).to(torch.int64)
    return x


def convertToTorchUint8(x):
    g = map(csr_matrix.toarray, x)
    x = torch.from_numpy(g).to_sparse()
    return x

def convertToCooSparse(x):
    t = []
    shape = x[0].shape
    for v in x:
        # values =
        # shape = v.shape
        # indices = v.indices
        v = v.tocoo()
        values = v.data
        indices = [v.row.tolist(), v.col.tolist()]
        i = torch.sparse_coo_tensor(indices, values=values, dtype=torch.int8, size=shape)
        # i = torch.
        t.append(i)

    t = torch.stack(t)
    return t
    t = torch.cat(t, dim=0)
    return t
    


def plot_perf(history, final_perf):
    epochs = range(1, len(history["loss"]) + 1)
    for key in ["loss", "accuracy", "precision", "recall", "f1"]:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history[key], "+-b", label=key)
        plt.plot(epochs, history["val_" + key], "+-g", label="val_" + key)
        plt.axhline(y=final_perf[key], color="r", linestyle="--", label="test_" + key)
        plt.legend()
        plt.title("Evolution of {} during training".format(key))
        plt.plot()


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def compute_loss(y, predictions, loss):
    # loss
    # binarize predictions from predictions (outputs = 1 if p>0.5 else 0)
    outputs = torch.argmax(predictions, axis=1)
    # metrics with accuracy, precision, recall, f1
    accuracy = accuracy_score(y.cpu(), outputs.cpu())
    precision, recall, f1 = [metric(y.cpu(), outputs.cpu(), average="micro") for metric in [precision_score, recall_score, f1_score]]
    return loss, accuracy, precision, recall, f1
    
def evaluate_loader(loader, model, criterion, device):
    # compute loss and accuracy for that loader
    metrics = {
        "loss": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }
    with torch.no_grad():
        # loop over examples of loader
        for i, data in enumerate(loader):

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            label_id = data["label_id"].to(device)
            logits, probs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, label_id)
            loss, accuracy, precision, recall, f1 = compute_loss(label_id, logits, loss)
            # sum up metrics in dict
            metrics["loss"] += loss.item()
            metrics["accuracy"] += accuracy
            metrics["precision"] += precision
            metrics["recall"] += recall
            metrics["f1"] += f1
        # normalize all values
        for k in metrics.keys():
            metrics[k] /= len(loader)
        return metrics

def show_cm(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, data in enumerate(loader):

            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            label_id = data["label_id"].to(device)
            now_batch_size = label_id.size(0)
            logits, probs = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.argmax(logits, axis=1)
            y_true.extend(label_id.tolist())
            y_pred.extend(outputs.tolist())
    cm = confusion_matrix(y_true, y_pred)
    return cm