import torch
import torch.nn as nn
import wandb
from torchvision.datasets import StanfordCars
from torchvision.models import resnet152
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from hparams import config
import torchvision.transforms as transforms
from timer import start_timer, end_timer_and_print
import json
import torch.profiler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result

def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Resize((224, 224)),
    ])
    train_dataset = StanfordCars('StanfordCars', transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    return train_loader

def prepare_model():
    model = resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])
    criterion = nn.CrossEntropyLoss()
    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    model.to(device)
    model.train()
    return model, criterion, optimizer, scaler

def train(train_loader):
    global model, criterion, optimizer
    images, labels = train_loader[0].to(device), train_loader[1].to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
def main():
    train_loader = data()
    model, criterion, optimizer, scaler = prepare_model()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18_batchsize1'),
            record_shapes=True,
            profile_memory=True,
            with_stack = True,
    ) as proof:
        for step, batch in enumerate(train_loader):
            if step >= (1 + 1 + 3) * 2:
                break
            train(batch)
            prof.step()
if __name__ == '__main__':
    main()