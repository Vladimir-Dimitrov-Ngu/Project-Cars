from torchvision.datasets import StanfordCars

if __name__ == "__main__":
    train_dataset = StanfordCars("StanfordCars/train", download=True)
    test_dataset = StanfordCars("StanfordCars/test", download=True)