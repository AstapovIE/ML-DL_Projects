import torch
import wandb
from model.Transformer import Transformer
from Training import LabelSmoothingLoss, NoamOpt, fit
from prepare_data import prepare_data


def train():
    # Загрузка данных
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO можно вынести prepare_data сюда в конечной версии
    train_iter, test_iter, word_field = prepare_data(DEVICE)

    # Создаем модель
    model = Transformer(vocab=word_field.vocab, d_model=256, device=DEVICE).to(DEVICE)

    pad_idx = word_field.vocab.stoi['<pad>']
    criterion_LB = LabelSmoothingLoss(vocab_size=len(word_field.vocab), padding_idx=pad_idx, smoothing=0.1).to(DEVICE)
    optimizer = NoamOpt(model)

    # Инициализация wandb
    wandb.init(project="TransformerSummarization", name="DVC_Training80")
    wandb.watch(model, log="all", log_freq=100)

    # Обучение модели
    trained_model, train_loss, val_loss = fit(model, criterion_LB, optimizer, train_iter, epochs_count=80, val_iter=test_iter, device=DEVICE, vocab=word_field.vocab)

    wandb.log({"Final Train Loss": train_loss, "Final Val Loss": val_loss})

    # Сохранение модели
    torch.save(model.state_dict(), "bert_100.pth")
    print("Модель сохранена в bert_100.pth")

    wandb.finish()

if __name__ == "__main__":
    train()
