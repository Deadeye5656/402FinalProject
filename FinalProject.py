import torchvision.models as models
import os
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import warnings
import numpy as np

warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FacesDataset(Dataset):
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

folder_path = "..\Celebrity Faces Dataset\Celebrity Faces Dataset"
celebrities = []

for root, dirs, files in os.walk(folder_path):
    for i, folder in enumerate(dirs):
        folder_path = os.path.join(root, folder)
        celebrity = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            celebrity.append(file_path)
        celebrities.append(celebrity)

FOLDTOTAL = 10
CELEBRITIESTOTAL = 17
kf = KFold(n_splits=FOLDTOTAL)
all_confusion_matrices = []
all_accuracies = []
for fold in range(FOLDTOTAL):
    test_file_paths = []
    test_labels = []
    train_labels = []
    train_file_paths = []
    alexnet = models.alexnet(pretrained=True)
    num_ftrs = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_ftrs, CELEBRITIESTOTAL)  # 17 classes

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

    for i, celebrity in enumerate(celebrities):
        fold_count = 0
        for train_index, test_index in kf.split(celebrity):
            if fold == fold_count:
                test_file_paths.extend([celebrity[i] for i in test_index])
                test_labels.extend([i for _ in test_index])
                fold_count += 1
            else:
                train_file_paths.extend([celebrity[i] for i in train_index])
                train_labels.extend([i for _ in train_index])
                fold_count += 1

        if i == CELEBRITIESTOTAL-1:
            train_dataset = FacesDataset(train_file_paths, train_labels, transform)

            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = alexnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            test_dataset = FacesDataset(test_file_paths, test_labels, transform)

            test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

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

            all_confusion_matrices.append(confusion_matrix(all_labels, all_preds))

            print('Fold: ' + str(fold+1))

            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

            all_accuracies.append(100 * correct / total)

print('Overall accuracy of the network: %d %%' % (sum(all_accuracies) / len(all_accuracies)))
print(np.sum(all_confusion_matrices, axis=0))

# Fold: 1
# Accuracy of the network on the test images: 91 %
# Fold: 2
# Accuracy of the network on the test images: 95 %
# Fold: 3
# Accuracy of the network on the test images: 96 %
# Fold: 4
# Accuracy of the network on the test images: 95 %
# Fold: 5
# Accuracy of the network on the test images: 97 %
# Fold: 6
# Accuracy of the network on the test images: 97 %
# Fold: 7
# Accuracy of the network on the test images: 96 %
# Fold: 8
# Accuracy of the network on the test images: 94 %
# Fold: 9
# Accuracy of the network on the test images: 95 %
# Fold: 10
# Accuracy of the network on the test images: 97 %
# Overall accuracy of the network: 95 %
# [[95  0  0  0  0  0  0  0  1  3  0  0  0  1  0  0  0]
#  [ 0 91  0  2  0  0  1  2  0  1  1  1  0  0  1  0  0]
#  [ 0  0 97  0  0  0  0  1  0  0  0  0  0  0  1  1  0]
#  [ 0  2  0 87  0  1  0  2  0  0  0  6  0  0  1  1  0]
#  [ 0  0  0  0 96  0  1  0  0  0  0  0  0  2  0  1  0]
#  [ 0  2  0  0  0 97  0  0  0  0  0  1  0  0  0  0  0]
#  [ 0  0  0  0  1  0 97  0  0  1  1  0  0  0  0  0  0]
#  [ 0  1  0  2  1  0  0 93  0  0  0  0  0  0  3  0  0]
#  [ 1  0  0  0  0  0  0  0 97  2  0  0  0  0  0  0  0]
#  [ 1  0  0  0  0  0  0  0  1 96  1  0  0  0  1  0  0]
#  [ 0  1  0  0  0  0  0  0  0  1 98  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0 99  0  1  0  0  0]
#  [ 0  0  0  0  0  0  0  0  1  1  0  0 98  0  0  0  0]
#  [ 0  0  0  0  1  0  1  0  0  0  1  1  0 96  0  0  0]
#  [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0 99  0  0]
#  [ 0  0  0  1  0  0  0  0  0  0  0  0  0  0  1 96  2]
#  [ 0  0  3  1  0  1  0  0  0  0  0  1  0  0  0  0 94]]
