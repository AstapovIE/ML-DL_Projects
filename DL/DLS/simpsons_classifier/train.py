import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fit_epoch(model, train_loader, criterion, optimizer, is_inception):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    progress_bar = tqdm(train_loader, desc=f"Train", dynamic_ncols=True)
    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        if is_inception:
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data

        progress_bar.set_postfix(loss=f"{train_loss:.4f}", acc=f"{train_acc:.4f}")

    return train_loss, train_acc


def eval_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    progress_bar = tqdm(val_loader, desc=f"Eval", dynamic_ncols=True)
    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)

        val_loss = running_loss / processed_size
        val_acc = running_corrects.double().cpu().numpy() / processed_size

        progress_bar.set_postfix(loss=f"{val_loss:.4f}", acc=f"{val_acc:.4f}")

    return val_loss, val_acc


def train(train_dataset, val_dataset, model, optimizer, criterion, epochs, batch_size, is_inception):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    history = []
    log_template = "Epoch {ep:03d}: train_loss: {t_loss:.4f}, val_loss: {v_loss:.4f}, train_acc: {t_acc:.4f}, val_acc: {v_acc:.4f}"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")  # Выводим номер эпохи без `tqdm`

        train_loss, train_acc = fit_epoch(model, train_loader, criterion, optimizer, is_inception)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)

        history.append((train_loss, train_acc, val_loss, val_acc))
        if val_acc >= 0.99:
            break

        tqdm.write(
            log_template.format(ep=epoch + 1, t_loss=train_loss, v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))

    return history, model

