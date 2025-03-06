from Masker import convert_batch
from SummaryGenerator import Generator

import math
from tqdm.auto import tqdm
from rouge import Rouge
import wandb

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

tqdm.get_lock().locks = []


class NoamOpt(object):
    def __init__(self, model, factor=2, warmup=4000, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model.d_model
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def do_epoch(model, criterion, data_iter, optimizer=None, name=None, device="cpu", epoch=0, generator=None):
    epoch_loss = 0
    is_train = optimizer is not None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    # Список для ROUGE-метрик (если валидация)
    rouge_2_scores = []
    rouge = Rouge()

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count, dynamic_ncols=True, leave=True, desc=name) as progress_bar:
            for i, batch in enumerate(data_iter):
                if i ==200:break
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch, device)
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                perplexity = math.exp(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "PPX": f"{perplexity:.2f}"})

                # Логируем каждые 500 батчей
                if i % 500 == 0:
                    wandb.log({"Batch Loss": loss.item()})

                # Для валидации считаем ROUGE на 5 примерах (только на одном батче)
                if not is_train and i == 0 and generator is not None:
                    source_text, target_text, output_text = generator.generate_summary(
                        source_inputs[:5], target_inputs[:5], source_mask[:5], target_mask[:5]
                    )

                    # Вычисляем средний ROUGE
                    scores = rouge.get_scores(output_text, target_text, avg=True)
                    rouge_2_scores.append(scores["rouge-2"]["f"])

            final_loss = epoch_loss / batches_count
            final_perplexity = math.exp(final_loss)

            progress_bar.set_postfix({"Final Loss": f"{final_loss:.5f}", "Final PPX": f"{final_perplexity:.2f}"})
            progress_bar.refresh()

    # Если валидация, логируем ROUGE
    if not is_train and generator is not None:
        avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0
        wandb.log({"ROUGE-2": avg_rouge_2})

    return final_loss


# def do_epoch(model, criterion, data_iter, optimizer=None, name=None, device="cpu", epoch=0):
#     epoch_loss = 0
#     is_train = optimizer is not None
#     name = name or ''
#     model.train(is_train)
#
#     batches_count = len(data_iter)
#
#     with torch.autograd.set_grad_enabled(is_train):
#         with tqdm(total=batches_count, dynamic_ncols=True, leave=True, desc=name) as progress_bar:
#             for i, batch in enumerate(data_iter):
#                 # if i == 100:break
#                 source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch, device)
#                 logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])
#
#                 logits = logits.contiguous().view(-1, logits.shape[-1])
#                 target = target_inputs[:, 1:].contiguous().view(-1)
#                 loss = criterion(logits, target)
#
#                 epoch_loss += loss.item()
#
#                 if optimizer:
#                     optimizer.optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                 perplexity = math.exp(loss.item())
#                 progress_bar.update(1)
#                 progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "PPX": f"{perplexity:.2f}"})
#
#                 # Логирование в wandb
#                 if i % 500 == 0:
#                     wandb.log({
#                         "Batch Loss": loss.item(),
#                         # "Batch Perplexity": perplexity
#                     })
#
#             final_loss = epoch_loss / batches_count
#             final_perplexity = math.exp(final_loss)
#             progress_bar.set_postfix({"Final Loss": f"{final_loss:.5f}", "Final PPX": f"{final_perplexity:.2f}"})
#             progress_bar.refresh()
#
#     return final_loss


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None, device="cpu", vocab=None):
    # Списки для сохранения потерь на обучении и валидации
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs_count+1):
        name_prefix = f'[{epoch} / {epochs_count}] '

        # Обучение модели на текущей эпохе
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:', device, epoch)
        train_losses.append(train_loss)

        if val_iter is not None:
            # Валидация модели на текущей эпохе
            generator = Generator(model=model, vocab=vocab)
            val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:', device, epoch, generator)
            val_losses.append(val_loss)
        else:
            val_loss = None

        # Логирование метрик после каждой эпохи
        # wandb.log({
        #     "Train Loss": train_loss,
        #     "Validation Loss": val_loss if val_loss is not None else 0,  # На случай, если нет валидации
        # })
        wandb.log({
                "Train/Val loss": wandb.plot.line_series(
                    xs=[i for i in range(epoch)],
                    ys=[train_losses, val_losses],
                    keys=["train", "val"],
                    title="Train/Val loss",
                    xname="epoch",)
            })

    return model, train_losses, val_losses


# def do_epoch(model, criterion, data_iter, optimizer=None, name=None, device="cpu"):
#     epoch_loss = 0
#
#     is_train = not optimizer is None
#     name = name or ''
#     model.train(is_train)
#
#     batches_count = len(data_iter)
#
#     with torch.autograd.set_grad_enabled(is_train):
#         with tqdm(total=batches_count, dynamic_ncols=True, leave=True, desc=name) as progress_bar:
#             for i, batch in enumerate(data_iter):
#                 source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch, device)
#                 logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])
#
#                 logits = logits.contiguous().view(-1, logits.shape[-1])
#                 target = target_inputs[:, 1:].contiguous().view(-1)
#                 loss = criterion(logits, target)
#
#                 epoch_loss += loss.item()
#
#                 if optimizer:
#                     optimizer.optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                 progress_bar.update(1)  # Правильное обновление прогресса
#                 progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "PPX": f"{math.exp(loss.item()):.2f}"})
#
#                 # Итоговые значения после эпохи
#             progress_bar.set_postfix({"Final Loss": f"{epoch_loss / batches_count:.5f}",
#                                       "Final PPX": f"{math.exp(epoch_loss / batches_count):.2f}"})
#             progress_bar.refresh()
#
#     return epoch_loss / batches_count
#
#
# def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None, device="cpu"):
#     for epoch in range(epochs_count):
#         name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
#         train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:', device)
#
#         if not val_iter is None:
#             val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:', device)
#
#     return model, train_loss, val_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        """
        vocab_size: размер словаря (количество классов)
        padding_idx: индекс <pad>, который не участвует в вычислении ошибки
        smoothing: коэффициент сглаживания (обычно 0.1)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing  # вероятность истинного класса

    def forward(self, logits, target):
        """
        logits: (batch_size * seq_len, vocab_size) - выход модели
        target: (batch_size * seq_len) - индексы истинных слов
        """
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))  # Распределяем вероятность на все классы
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # Истинному классу даем больше веса
            true_dist[:, self.padding_idx] = 0  # На паддинги вероятность не распределяем

            mask = (target == self.padding_idx)  # Убираем вклад <pad> в лосс
            true_dist[mask] = 0

        return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=-1), dim=-1))
