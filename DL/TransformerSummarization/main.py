import torch
import torch.nn as nn
import wandb  # Подключаем wandb

from prepare_data import prepare_data
from Transformer import Transformer
from Training import LabelSmoothingLoss, NoamOpt, fit
from SummaryGenerator import Generator
from hometasks_functions import task1, task3



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

train_iter, test_iter, word_field = prepare_data(DEVICE)

model = Transformer(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab), d_model=256).to(DEVICE)

pad_idx = word_field.vocab.stoi['<pad>']
criterion_LB = LabelSmoothingLoss(vocab_size=len(word_field.vocab), padding_idx=pad_idx, smoothing=0.1).to(DEVICE)

optimizer = NoamOpt(model)

# Инициализация wandb
wandb.init(project="TransformerSummarization", name="Test1")
# Подключаем модель к wandb для логирования градиентов и параметров
wandb.watch(model, log="all", log_freq=100)

# Обучение
trained_model, train_loss, val_loss = fit(model, criterion_LB, optimizer, train_iter, epochs_count=4, val_iter=test_iter, device=DEVICE, vocab=word_field.vocab)

# Логируем финальные значения после обучения
wandb.log({"Final Train Loss": train_loss, "Final Val Loss": val_loss})

print("train_loss", train_loss, "val_loss", val_loss)

# Сохранение модели
torch.save(model.state_dict(), "transformer_model.pth")
print("Модель сохранена в файл transformer_model.pth")

# Логируем сохраненную модель в wandb
# wandb.save("transformer_model.pth")

# Загружаем веса
# model.load_state_dict(torch.load("transformer_model.pth", map_location=DEVICE))
# model.eval()
# print("Модель загружена!")

# Генерация суммаризаций и анализ attention
# generator = Generator(model=model, vocab=word_field.vocab)
# task1(generator, test_iter, 5, DEVICE, "demo_result")
# task3(model, test_iter, word_field.vocab, 3, DEVICE, "attention")

# Завершаем логирование
wandb.finish()
