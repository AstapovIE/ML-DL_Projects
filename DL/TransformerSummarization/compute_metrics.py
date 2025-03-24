import torch
from model.Transformer import Transformer
from model.SummaryGenerator import Generator
from hometasks_functions import task1, task3
from prepare_data import prepare_data


def compute_metrics():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем данные и модель
    train_iter, test_iter, word_field = prepare_data(DEVICE)
    model = Transformer(vocab=word_field.vocab, d_model=256, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("bert_100.pth", map_location=DEVICE))
    model.eval()

    # Генерация тестовых суммаризаций
    generator = Generator(model=model)
    task1(generator, test_iter, 5, DEVICE, "data/demo_result")
    task3(model, test_iter, word_field.vocab, 3, DEVICE, "data/attention")

if __name__ == "__main__":
    compute_metrics()