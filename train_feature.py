import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder


class Dataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

class TrainFaceNet:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(
            classify=False,
            num_classes=1280
        ).to(self.device)
        self.model.train()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_model(self, dataset, max_epochs=10, batch_size=32):
        for epoch in range(max_epochs):
            train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            for idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(data) #RuntimeError here
                #RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[11, 512, 512, 3] to have 3 channels, but got 512 channels instead

                output = output.view(output.size(0), -1)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if (idx + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{max_epochs}], Step [{idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

#read all images from folder train_faceid train and create dataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x_train = []
y_train = []
model = InceptionResnetV1(
            classify=False,
            num_classes=1280
        ).to(device)
model.train()

for idx, path in enumerate(os.listdir('train_faceid')):
    img = cv2.imread(os.path.join('train_faceid', path))
    img = cv2.resize(img, (512, 512))

    
    img = fixed_image_standardization(img)
    img = torch.from_numpy(np.array(img, dtype=np.float32)).to(device)

    x_train.append(img)
    y_train.append(path.split('.')[0])
    
    if idx == 10:
        break

#one hot encodding for y_train to vetor 1280
le = LabelEncoder()
y_train = le.fit_transform(y_train)

dataset = Dataset(x_train, y_train)

train = TrainFaceNet()
model = train.train_model(dataset)
