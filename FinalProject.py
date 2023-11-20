import torchvision.models as models
import os
import cv2
from PIL import Image
import numpy as np

alexnet = models.alexnet(pretrained=True)

folder_path = "..\Celebrity Faces Dataset\Celebrity Faces Dataset"
celebrities = []
for root, dirs, files in os.walk(folder_path):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        celebrity = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            celebrity.append(file_path)
        celebrities.append(celebrity)

#define model
for foldIndex in range(0, 10):
    for celebrity in celebrities:
        testList = []
        # define label
        for i in range(0,100):
            if i >= (foldIndex*10) and i < ((foldIndex+1)*10):
                testList.append(celebrity[i])
            else:
                img = cv2.imread(celebrity[i])
                if img is not None: 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array = Image.fromarray(img, 'RGB')
                    img_rs = img_array.resize((227,227))
                    img_rs = np.array(img_rs)
                print(celebrity[i])
        # test accuracy