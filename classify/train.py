import torch
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils import *
from model import ResNet9


def main():
    train_tfms = tt.Compose(
        [tt.Resize((32, 32)), tt.ToTensor()]
    )

    test_tfms = tt.Compose([tt.Resize((32, 32)), tt.ToTensor()])

    train_set = MNIST(
        root="./datasets", train=True, download=True, transform=train_tfms
    )
    test_set = MNIST(root="./datasets", train=False, download=True, transform=test_tfms)

    device = get_default_device()

    train_loader = DeviceDataLoader(
        DataLoader(train_set, shuffle=True, batch_size=16), device
    )
    test_loader = DeviceDataLoader(
        DataLoader(test_set, shuffle=False, batch_size=16), device
    )

    # Defining model and training options
    model = ResNet9(1, 10)
    to_device(model, device)

    epochs = 30
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    
    history = [evaluate(model, test_loader)]
  
    print("Initial history: ", history)

    print("Using device: ", device,
          f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "cpu")
    
    print("Starting training....")
    history += fit_one_cycle(
        epochs,
        max_lr,
        model,
        train_loader,
        test_loader,
        "./checkpoints_30",
        False,
        weight_decay,
        grad_clip,
        opt_func,
    )

    plot_metrics(history)

if __name__ == "__main__":
    main()
