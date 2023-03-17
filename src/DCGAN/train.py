import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from model import Disc, Generator, initialize_weights
import sys
sys.path.append('/Users/arihanttripathi/Documents/AutoVariateGithub')
from AutoVariate.utils import util
import os


util_class = util.auto_util()

class Train:
    
    def __init__(self, lr=2e-4, batch_size=128, image_size=64, channels=1, features_disc=64, features_gen=64, z_dim=100, num_epochs=10):
        self.lr = lr
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.features_disc = features_disc
        self.features_gen = features_gen
        self.z_dim = z_dim
        self.name = os.path.basename(__file__)
        self.num_epochs = num_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def transform_set(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(self.channels)], [0.5 for _ in range(self.channels)]
            ),
        ])
        return transform
    
    def set_dataset(self, dataset=None):
        self.name = str(util_class.list_sub_dir("dataset/")[0])
        if dataset == None:
            dataset = datasets.MNIST(root="dataset/", train=True, transform=self.transform_set(), download=True)
        else:
            self.dataset = dataset
        return self.name

def main():
    t = Train()
    dataset = t.set_dataset()
    print(dataset)
    # util_class.erase_dir("dataset")

main()