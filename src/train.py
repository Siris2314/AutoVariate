import torch
from torch import nn
from torch.nn import functional as F
from src.model import AutoVariate
from torchvision.utils import save_image
import torch
from tqdm import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import cpuinfo
from torch.utils.data import DataLoader
import warnings
from src.logger import Auto_Var_Logger

log = Auto_Var_Logger('WARNING')

class auto_variate():
    def __init__(self, input_dim=0, hidden_dim=0, z_dim=0, lr_rate=0, batch_size=0, num_cpu=0, epochs=0, dataset = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.num_cpu = num_cpu
        self.epochs = epochs
        self.dataset = dataset


        if self.dataset == None:
            log.log("Dataset not found, using MNIST")
        if self.input_dim == 0:
            log.log("You are using the default dimensions, which are all set to 0, it's pretty much pointless")
        if self.hidden_dim == 0:
            log.log("You are using the default dimensions, which are all set to 0, it's pretty much pointless")

            

    def create_model(self,input_dim, hidden_dim, z_dim, device):
        model = AutoVariate(input_dim, hidden_dim, z_dim).to(device)
        return model

    def get_values(self):
        return self.input_dim, self.hidden_dim, self.z_dim, self.lr_rate, self.batch_size, self.num_cpu, self.epochs

    def set_dataloader(self):
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpu)
        return dataloader

    def set_dataset(self,dataset=None, train=True, transform=transforms.ToTensor(), download=True):
        if(dataset == None):
            self.dataset = datasets.MNIST(root='dataset/', train=train, transform=transform, download=download)  
        self.dataset = dataset


    def get_dataset(self):
        return self.dataset

    def return_device_training(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def return_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(str(device) == 'cpu'):
            return cpuinfo.get_cpu_info()['brand_raw']
        return torch.cuda.get_device_name(0)

    def loading_train(self,train_loader):
        loop = tqdm(enumerate(train_loader))
        return loop


    def create_optimizer(self,lr_rate, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        return optimizer

    def loss_function(self):
        loss = nn.BCELoss(reduction='sum')
        return loss
    
    def save_model(self, model, name='model'):
        torch.save(model.state_dict(), name +'.pth')


    def train(model, dataloader, optimizer, loss, device):
        for epoch in range(at.epochs):
            loop = at.loading_train(at.set_dataloader())
            for i, (x, _) in loop:
                x = x.to(device).view(x.shape[0],at.input_dim)
                x_reconstructed, mu, sigma = model(x)

                reconstruction_loss = loss(x_reconstructed, x)
                kl_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                total_loss = reconstruction_loss + kl_divergence
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                loop.set_postfix(loss=total_loss.item())

#Example of how to use the class
if __name__ == '__main__':
     at = auto_variate()
     at.set_dataset()
     device = at.return_device_training()
     model = at.create_model(at.input_dim, at.hidden_dim, at.z_dim, device).to(device)
     optimizer = at.create_optimizer(at.lr_rate, model)
     loss = at.loss_function()
     at.train(model, at.set_dataloader(), optimizer, loss, device)
     at.save_model(model)














