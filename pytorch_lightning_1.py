# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch modules
import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torchmetrics


# Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

class Cifar10_jm_model(LightningModule):
    
    def __init__(self,lr=1e-3):
        '''method used to define our model parameters'''
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        # metrics
        #self.accuracy = accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        batch_size, channels, width, height = x.size()

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log training loss
        self.log('train_loss', loss)

        # Log metrics
        self.log('train_acc', torchmetrics.functional.accuracy(logits, y))

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # Log metrics
        self.log('valid_acc', torchmetrics.functional.accuracy(logits, y))

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log test loss
        self.log('test_loss', loss)

        # Log metrics
        self.log('test_acc', torchmetrics.functional.accuracy(logits, y))
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
    
    
class Cifar10_datamodule(LightningDataModule):
    
    def __init__(self, data_dir='./', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            
            train_size = int(0.8 * len(cifar10_train))
            test_size = len(cifar10_train) - train_size
            
            self.cifar10_train, self.cifar10_val = random_split(cifar10_train, [train_size, test_size])
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar10_train = DataLoader(self.cifar10_train, batch_size=self.batch_size)
        return cifar10_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar10_val = DataLoader(self.cifar10_val, batch_size=self.batch_size)
        return cifar10_val

    def test_dataloader(self):
        '''returns test dataloader'''
        cifar10_test = DataLoader(self.cifar10_test, batch_size=self.batch_size)
        return cifar10_test
    
    
wandb.login()
wandb_logger = WandbLogger(project='CIFAR10_JM')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
# setup data
cifar10 = Cifar10_datamodule()
 
# setup model - choose different hyperparameters per experiment
model = Cifar10_jm_model(lr=1e-3)

trainer = Trainer(
    logger=wandb_logger,    # W&B integration
    accelerator="cpu",                # use all GPU's
    max_epochs=3            # number of epochs
    )

trainer.fit(model, cifar10)
trainer.test(model, datamodule=cifar10)
wandb.finish()