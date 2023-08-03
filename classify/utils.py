import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from tqdm import trange, tqdm



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        
@torch.no_grad()
def evaluate(model, val_loader):
    print("Evaluating model")
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def get_latest_checkpoint(checkpoint_directory):
    all_files = os.listdir(checkpoint_directory)

    checkpoint_files = [file for file in all_files if file.startswith("checkpoint_") and file.endswith(".pth")]

    if not checkpoint_files:
        return None 

    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split("_")[1][:-4]))

    latest_checkpoint = sorted_checkpoints[-1]
    latest_checkpoint_path = os.path.join(checkpoint_directory, latest_checkpoint)

    return latest_checkpoint_path

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  checkpoint_path=None, resume_training=False,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))    

    if checkpoint_path and resume_training:
        print("Loading checkpoint....")
        checkpoint = torch.load(get_latest_checkpoint(checkpoint_path))
        model_state = checkpoint['model_state_dict']
        model.load_state_dict(model_state)
        optimizer_state = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint['scheduler_state_dict']
        sched.load_state_dict(scheduler_state)
        sched.last_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch'] + 1
        print("Successfully loaded checkpoint")
    else:
        start_epoch = 0


    
    for epoch in trange(start_epoch, epochs, desc="Training"):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
        # Save checkpoint
        if checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': sched.state_dict()
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, "checkpoint_{}.pth".format(epoch + 1)))
    
    return history

def calculate_mean_std(dataset, image_size=(28, 28)):
    # Define transformations to resize and convert images to tensors
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Create a DataLoader with batch_size=1 to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    num_channels = 1 

    for inputs, _ in data_loader:
        num_channels = inputs.size(1)
        break
    
    mean = torch.zeros(num_channels)  
    std = torch.zeros(num_channels)
    num_samples = len(data_loader.dataset)
    for inputs, _ in data_loader:
        mean += inputs.squeeze().mean(1).mean(1)  # Calculate the mean along height and width axes (H and W)
        std += inputs.squeeze().std(1).std(1)  # Calculate the standard deviation along height and width axes (H and W)

    mean /= num_samples
    std /= num_samples

    return mean.tolist(), std.tolist()

def plot_metrics(metrics_list):
    # Extract the values for each metric from the list of dictionaries
    epochs = list(range(1, len(metrics_list) + 1))
    val_losses = [entry["val_loss"] for entry in metrics_list]
    val_accs = [entry["val_acc"] for entry in metrics_list]

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def predict(model, sample, classes, device):
    img = sample[0].to(device)
    label = sample[1]
    prediction = model(img.unsqueeze(0))
    print("Label: {}".format(classes[label]))
    print("Prediction: {}".format(classes[torch.argmax(prediction.squeeze(0))]))
    plt.imshow(img.detach().squeeze(0).cpu().numpy(), cmap = "gray")
    plt.axis('off')
    plt.show()
    