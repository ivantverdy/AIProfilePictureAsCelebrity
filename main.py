import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as trans

# import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/ashwingupta3012/human-faces'
# od.download(dataset_url)

dataset_dir = './human-faces'

image_size = 640
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = ImageFolder(dataset_dir, transform=trans.Compose(
    [trans.Resize(image_size), trans.CenterCrop(image_size), trans.ToTensor(), trans.Normalize(*stats)]))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)