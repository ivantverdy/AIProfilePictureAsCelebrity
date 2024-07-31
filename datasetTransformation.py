import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.Resize((640, 640)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

human_faces_dataset = datasets.ImageFolder(root='human-faces/Humans', transform=transform)
celeba_dataset = datasets.ImageFolder(root='./celeba-dataset/img_align_celeba/img_align_celeba', transform=transform)

batch_size = 128

human_faces_loader = torch.utils.data.DataLoader(human_faces_dataset, batch_size=batch_size, shuffle=True)
celeba_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)