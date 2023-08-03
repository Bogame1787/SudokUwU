import torch
from utils import get_latest_checkpoint, to_device, predict
from model import ResNet9
from torchvision.datasets import MNIST
import torchvision.transforms as tt

checkpoint_path = "./checkpoints_30/"
checkpoint = torch.load(get_latest_checkpoint(checkpoint_path))

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet9(1, 10)

model_state = checkpoint['model_state_dict']
model.load_state_dict(model_state)

to_device(model, device)

test_tfms = tt.Compose([tt.Resize((32, 32)), tt.ToTensor()])
test_set = MNIST(root="./datasets", train=False, download=True, transform=test_tfms)

sample_img = test_set[9000]

predict(model, sample_img, test_set.classes, device)