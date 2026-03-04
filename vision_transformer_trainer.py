import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from vision_transformer import VisionTransformer

class VisionTransformerTrainer():
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

        self.trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,  
            shuffle=True, 
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )

        self.classes = self.dataset.classes

        self.model = VisionTransformer()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_rate)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        if save_path is None:
            self.save_path = os.path.join(os.getcwd(), "vision_transformer.pth")
        else:
            if os.path.isdir(save_path) or str(save_path).endswith(os.sep):
                os.makedirs(save_path, exist_ok=True)
                self.save_path = os.path.join(save_path, "vision_transformer.pth")
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

    def train(self):
        best_val_acc = max(self.val_accuracies) if self.val_accuracies else 0.0
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            avg_loss = running_loss / len(self.trainloader)
            train_acc = 100 * correct_train / total_train
            val_acc = self.validate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                print(f"New best model saved with Validation Accuracy: {val_acc:.2f}%")

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = running_val_loss / len(self.valloader)
        self.val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        return accuracy

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