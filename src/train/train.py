import numpy as np
from tqdm import tqdm

def train(model, train_loader):  # dataset/watch_log.py : get_dataset() -> train_dataset, valid_dataset, test_dataset
                                 # train_loader = SimpleDataLoader(train_dataset, ...)
    total_loss = 0
    for features, labels in tqdm(train_loader):  # batch_size=32  if len:100 -> 32 32 32 4
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)
        loss = np.mean((predictions - labels) ** 2)
        model.backward(features, labels, predictions)
        total_loss += loss

    return total_loss / len(train_loader)