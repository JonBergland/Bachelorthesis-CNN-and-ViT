import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class BaseTrainer:
    def __init__(self, 
                 dataset_root: str, 
                 epochs: int = 5,
                 lr_rate: float = 0.01,
                 batch_size: int = 32,
                 img_size: int = 32, 
                 manual_seed: int = 42,
                 save_path: str | None = None,
                 only_see_metrics: bool = False):

        self.epochs = epochs
        self.batch_size = batch_size

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(), # Data augmentations
            # transforms.RandomRotation(10),     # Data augmentation
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226])
        ])

        self.generator = torch.Generator().manual_seed(manual_seed)

        self.dataset = torchvision.datasets.ImageFolder(root=dataset_root, transform=transform)

        n = len(self.dataset)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size

        trainset, valset, testset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=self.generator
        )

        self.trainloader = DataLoader(
            trainset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.valloader = DataLoader(
            valset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.testloader = DataLoader(
            testset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.classes = self.dataset.classes

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), "model.pth")
        else:
            if os.path.isdir(save_path) or str(save_path).endswith(os.sep):
                os.makedirs(save_path, exist_ok=True)
                self.save_path = os.path.join(save_path, "model.pth")
            else:
                parent = os.path.dirname(save_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self.save_path = save_path

        if os.path.exists(self.save_path):
            try:
                self.load_model(self.save_path)
            except Exception as e:
                print(f"Warning: failed to load model from {self.save_path}: {e}")

        if only_see_metrics:
            self.plot_metrics()
            exit()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs_range = range(1, len(self.train_accuracies) + 1)

        ax1.plot(epochs_range, self.train_accuracies, label='Train Accuracy', marker='o')
        ax1.plot(epochs_range, self.val_accuracies, label='Validation Accuracy', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, self.train_losses, label='Training Loss', marker='o')
        ax2.plot(epochs_range, self.val_losses, label='Validation Loss', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training vs Validation Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()