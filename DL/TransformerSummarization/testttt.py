import torch

data = torch.load("processed_data/data.pth", weights_only=False)
print(type(data))  # Должно быть tuple или list
print(data)  # Посмотреть, что внутри