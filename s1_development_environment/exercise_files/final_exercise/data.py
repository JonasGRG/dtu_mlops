import torch
from torch.utils.data import Dataset, DataLoader


class CorruptMNISTDataset(Dataset):
    def __init__(self, image_files, target_files):
        # Load all images and targets
        self.images = [torch.load(file) for file in image_files]
        self.targets = [torch.load(file) for file in target_files]

        # Flatten the list of images and targets
        self.images = torch.cat(self.images, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
    

def mnist(batch_size = 64, num_workers=4):
    """Return train and test dataloaders for MNIST."""
    # Paths to your image and target files
    train_image_files = [f'../data/corruptmnist/train_images_{i}.pt' for i in range(1,6)]
    train_target_files = [f'../data/corruptmnist/train_target_{i}.pt' for i in range(1,6)]

    val_image_files = [f'../data/corruptmnist/train_images_0.pt']
    val_target_files = [f'../data/corruptmnist/train_target_0.pt']

    test_image_files = ['../data/corruptmnist/test_images.pt']
    test_target_files = [f'../data/corruptmnist/test_target.pt']

    train_dataset = CorruptMNISTDataset(train_image_files, train_target_files)
    val_dataset = CorruptMNISTDataset(val_image_files, val_target_files)
    test_dataset = CorruptMNISTDataset(test_image_files, test_target_files)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
        )
    
    return train_loader, val_loader, test_loader
