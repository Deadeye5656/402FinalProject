import torchvision.models as models
import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset
class CelebrityFacesDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

# Model
alexnet = models.alexnet(pretrained=True)
num_ftrs = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(num_ftrs, 17)  # 17 classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

# Data preparation
folder_path = "..\Celebrity Faces Dataset\Celebrity Faces Dataset"
celebrities = []

labels = []
for root, dirs, files in os.walk(folder_path):
    for i, folder in enumerate(dirs):
        folder_path = os.path.join(root, folder)
        celebrity = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            celebrity.append(file_path)
            labels.append(i)
        celebrities.append(celebrity)

for celebrity in celebrities:
    # LOOCV
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(celebrity):
        train_file_paths = [celebrity[i] for i in train_index]
        test_file_paths = [celebrity[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        train_dataset = CelebrityFacesDataset(train_file_paths, train_labels, transform)
        test_dataset = CelebrityFacesDataset(test_file_paths, test_labels, transform)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Training
        for epoch in range(2):  # loop over the dataset multiple times
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = alexnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Testing
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = alexnet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        print(train_dataset.file_paths[0].split('\\')[-2])

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        # Confusion matrix
        print(confusion_matrix(all_labels, all_preds))
