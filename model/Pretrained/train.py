
import torch
import torch.nn as nn
import wandb
from torchvision.datasets import StanfordCars
from torchvision.models import resnet152
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from hparams import config
import torchvision.transforms as transforms
# api = wandb.Api()
wandb.init(config=config, project="Car-project", name=config['name'])

def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result


def main():
    assert not (
            config['loss_scaling'] and config['precision'] == "half"
    ), "Cannot use standard loss scaler with FP16 model weights"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Resize((224, 224)),
    ])
    train_dataset = StanfordCars('StanfordCars', transform=transform, download=False)
    test_dataset = StanfordCars('StanfordCars', transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config['num_classes'])
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()
    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])


    i = 0
    if config['loss_scaling']:
        scaler = torch.cuda.amp.GradScaler()
    if config['precision'] == 'half':
        model.half()
    model.to(device)
    for epoch in trange(config["epochs"]):
        model.train()
        for images, labels in tqdm(train_loader, desc = 'Обучение', position = 0):
            if config['precision'] == 'half':
                images = images.half()
            i += 1
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            if config['loss_scaling']:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0 or epoch == config['epochs'] - 1:
            all_preds = []
            all_labels = []
            model.eval()
            for test_images, test_labels in tqdm(test_loader, desc = 'Тестирование', position = 0):
                if config['precision'] == 'half':
                    test_images = test_images.half()
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                with torch.inference_mode():
                    outputs = model(test_images)
                    preds = torch.argmax(outputs, 1)

                    all_preds.append(preds)
                    all_labels.append(test_labels)

            accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))

            metrics = {'test_acc': accuracy, 'train_loss': loss}
            wandb.log(metrics, step=epoch * len(train_dataset) + (i + 1) * config["batch_size"])
    torch.save(model.state_dict(), "model.pt")

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


if __name__ == '__main__':
    main()