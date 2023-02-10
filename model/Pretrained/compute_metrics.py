import json
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from torchvision.datasets import StanfordCars
from torchvision.models import resnet152

from hparams import config


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Resize((224, 224))
    ])

    test_dataset = StanfordCars(root='StanfordCars',
                           transform=transform,
                           download=False,
                           )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config["batch_size"])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = resnet152(pretrained=False, num_classes=config['num_classes'])
    model.load_state_dict(torch.load("model.pt"))
    model.to(device)

    correct = 0.0

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        if config['precision'] == 'half':
            test_images = test_images.half()
        with torch.inference_mode():
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)
            correct += (preds == test_labels).sum()

    accuracy = correct / len(test_dataset)

    with open("final_metrics.json", "w+") as f:
        json.dump({"accuracy": accuracy.item()}, f)
        print("\n", file=f)

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args('')
    main(args)