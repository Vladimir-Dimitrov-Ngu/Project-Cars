from torchvision.datasets import StanfordCars
from IPython.display import clear_output
if __name__ == "__main__":
    train_dataset = StanfordCars("StanfordCars", download=True)
    test_dataset = StanfordCars("StanfordCars", download=True)
    clear_output()
    print('Done! Data uploaded')