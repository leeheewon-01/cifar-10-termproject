import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, Normalize, ToTensor, AugMix, CenterCrop
from torchvision.datasets import CIFAR10
from utils.loss import ASLSingleLabel

import timm

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

CFG = {
    'Learning_rate': 1e-4,
    'EPOCHS': 100,
    'BATCH_SIZE': 16,
    'Note': 'baselinev4, SGD, Step, ASL, imgsz 224, bz 16',
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

mean= [0.4914, 0.4822, 0.4465]
std= [0.2023, 0.1994, 0.2010]

composed_train = Compose([Resize(256),
                            RandomCrop(224),
                            RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                            AugMix(),
                            ToTensor(),
                            Normalize(mean, std)
                            ])

composed_test = Compose([Resize(256),
                           CenterCrop(244),
                           ToTensor(),
                           Normalize(mean, std)
                           ])

root_dir = "../data/CIFAR_10"

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)

    criterion = ASLSingleLabel().to(device)
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        total, correct = 0, 0
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        train_acc = correct / total

        print(f'Epoch [{epoch}] Train Loss : [{_train_loss:.5f}] Train Acc [{train_acc:.5f}] Val Loss : [{_val_loss:.5f}] Val Acc : [{_val_score:.5f}]')
        
        # scheduler with warmup
        scheduler.step()
        
        wandb.log({"Val Loss": _val_loss,
                   "Val Acc": _val_score,
                   "Train Loss": _train_loss,
                   "Train Acc": train_acc,
                   "lr" : optimizer.param_groups[0]['lr']})
        
    return model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    totals, corrects = 0, 0

    with torch.no_grad():
        for imgs, labels in iter(val_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            pred = model(imgs)
            loss = criterion(pred, labels)
            val_loss.append(loss.item())

            _, predicted = pred.max(1)
            totals += labels.size(0)
            corrects += predicted.eq(labels).sum().item()
        
        _val_loss = np.mean(val_loss)
        _val_score = corrects / totals
    
    return _val_loss, _val_score

model_num = 10

for s in range(model_num):
    os.environ['WANDB_START_METHOD'] = 'thread'

    run_name = f'ASL_baselinev4_16_seed_{s}'
    wandb.init(project="cifa10_proj_baseline", name=run_name)

    seed_everything(s)

    wandb.config.update(CFG)

    train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=composed_train)
    test_dataset = CIFAR10(root=root_dir, train=False, transform=composed_test)

    train_loader_dict = {'batch_size': CFG['BATCH_SIZE'], 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
    test_loader_dict = {'batch_size': 100, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    train_loader = DataLoader(dataset=train_dataset, **train_loader_dict)
    test_loader = DataLoader(dataset=test_dataset, **test_loader_dict)
    
    model = timm.create_model('hf_hub:timm/resnet18.fb_swsl_ig1b_ft_in1k', num_classes=10)
    model = nn.DataParallel(model, device_ids=[0, 1], dim=0, output_device=0)
    model.load_state_dict(torch.load('./Models/model_9741.pt'))

    optimizer = torch.optim.SGD(model.parameters(), lr=CFG['Learning_rate'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    infer_model = train(model, optimizer, train_loader, test_loader, exp_lr_scheduler, device)

    torch.save(infer_model.state_dict(), f'./baselinev4_models_seed_{s}.pt')

    wandb.finish()

    del infer_model, model, optimizer, exp_lr_scheduler, train_loader, test_loader, train_dataset, test_dataset
