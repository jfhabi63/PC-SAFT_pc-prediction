import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import *
from model_BNN import *
from data_extraction import *


build_dir = Path("build")
if not build_dir.exists():
    build_dir.mkdir()

train_fraction = 0.8
dataset = NoiseDataset()

train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(2)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
loader_kwargs = {
    "batch_size": 64,
    "drop_last": False,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
}
train_loader = DataLoader(dataset=train_dataset,
                          **loader_kwargs
                          )
test_loader = DataLoader(dataset=test_dataset,
                         **loader_kwargs
                         )

net = NN().to(device)
