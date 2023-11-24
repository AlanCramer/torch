
import torch
import torch.nn as nn
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import pytorch_model_summary
import time
import torchvision
     
dataDir = Path('.', 'dataset')
checkpointDir = Path('.', 'checkpoints', 'fam')

print(dataDir)
print(checkpointDir)

dataDir.mkdir(exist_ok=True, parents=True)
checkpointDir.mkdir(exist_ok=True, parents=True)

fileDir = Path(dataDir, 'mnist')
qFileName = {}
qFiles = os.listdir(fileDir)
for fName in qFiles:
    fNameTmp = fName.lower()
    if fNameTmp.find('train') != -1:
        qFileName['train'] = fName
    elif fNameTmp.find('test') != -1:
        qFileName['test'] = fName

torch.random.manual_seed(42)
np.random.seed(42)

print(qFileName)

def convBlock(inp, oup, kernel, stride, padding, bias=True, groups=1):
    return nn.Sequential(
      nn.Conv2d(inp, oup, kernel, stride, padding, bias=bias, groups=groups),
      nn.BatchNorm2d(oup),
    )

class SqueezeExciteLayer(nn.Module):
    def __init__(self, outDim, reductionNum):
        super(SqueezeExciteLayer, self).__init__()
        hiddenNum = int(outDim / reductionNum)
        self.fc1 = nn.Linear(outDim, hiddenNum)
        self.fc2 = nn.Linear(hiddenNum, outDim)
        self.outDim = outDim

    def forward(self, x):
        squeeze = x.mean(3).mean(2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = torch.reshape(excitation, [-1, self.outDim, 1, 1])
        return x * excitation

class InvertedResidualv3(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, bias=True, expandRatio=6, reductionNum=4):
        super(InvertedResidualv3, self).__init__()
        self.stride = stride
        self.inp = inp
        self.oup = oup
        hiddenDim = int(inp * expandRatio)
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []        
        if expandRatio == 1:
            # depthwise
            layers.append(convBlock(inp, hiddenDim, kernel, stride, padding, groups=hiddenDim, bias=bias))
            layers.append(nn.LeakyReLU())
            # pointwise
            layers.append(convBlock(hiddenDim, oup, 1, 1, padding, bias=bias))
        else:
            # pointwise
            layers.append(convBlock(inp, hiddenDim, 1, 1, padding, bias=bias))
            layers.append(nn.LeakyReLU())
            # depthwise
            layers.append(convBlock(hiddenDim, hiddenDim, kernel, self.stride, 1, groups=hiddenDim, bias=bias))
            layers.append(nn.LeakyReLU())
            # squeeze&excite layer
            layers.append(SqueezeExciteLayer(hiddenDim, reductionNum))
            # pointwise
            layers.append(convBlock(hiddenDim, oup, 1, 1, padding, bias=bias))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, inDim=2, outDim=19):
        super(MobileNetV3, self).__init__()
        block = InvertedResidualv3
        bias = False
        kernel = (3,3)
        inputChannel = 12
        expandRatio = 6
        padding='valid'
        invertedResidualSetting = [
          # t, c, n, s
          [6, 12, 2, 2],
          [6, 24, 1, 2],
        ]

        self.features = [convBlock(inDim, inputChannel, kernel, 1, padding, bias=bias)]
        for t, c, n, s in invertedResidualSetting:
            outputChannel = c
            for i in range(n):
                if i == 0:
                    self.features.append(block(inputChannel, outputChannel, kernel, s, padding, bias, expandRatio=t))
                else:
                    self.features.append(block(inputChannel, outputChannel, kernel, 1, padding, bias, expandRatio=t))
                inputChannel = outputChannel
        self.features = nn.Sequential(*self.features)
        self.conv1 = nn.Sequential(
            convBlock(inputChannel, inputChannel * expandRatio, 1, 1, padding),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inputChannel * expandRatio, inputChannel, 1, 1, padding),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(inputChannel, outDim)
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        x = self.conv2(x)
        return F.softmax(self.fc1(torch.squeeze(x)), dim=1)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.t(m.weight, gain=0.1)
                nn.init.normal_(m.weight, std=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
     
class DigitDataset(Dataset):
    def __init__(self, csv_data, transform, phase='training'):
        self.csv_data = csv_data
        self.transform = transform
        raw_tensor = torch.tensor(csv_data)
        self.phase = phase
        if self.phase == 'training':
            self.data, self.labels = raw_tensor[:, 1:].view(-1, 28, 28), raw_tensor[:, 0]
        if self.phase == 'testing':
            # test dataset does not have label, only digit data
            self.data = raw_tensor.view(-1, 28, 28)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.transform(self.data[idx]), self.labels[idx]
        if self.phase == 'testing':
            # idx+1 is not used, but it is real label of the test dataset
            return self.transform(self.data[idx]), (idx+1)
        
def training(model, datasets, datasets_size, optimizer, scheduler, epochs=10):
    # Conduct train and valid per a epoch
    phases = ['train', 'valid']
    
    best_valid_accuracy = -1.0
    
    training_loss = []
    validation_loss = []
        
    for epoch in range(epochs):                
        print(f'-------------- Epoch {epoch+1} --------------')        
        
        for phase in phases:
            epoch_begin = time.time()
            epoch_accuracy = 0.0
            epoch_losses = 0.0
            
            if phase == 'train':
                print(" about to train ")
                model.train()
                print(" done train ")
                
            if phase == 'valid':
                print(" about to valid")
                model.eval()
                print(" done valid")
                
            for data in datasets[phase]:
                # Init weights for each batch dataset
                if phase == 'train':
                    optimizer.zero_grad()
                    
                inputs, labels = data
                # to use nll_loss, labels should be LongTensor
                labels = labels.type(torch.LongTensor)
                if torch.cuda.is_available():
                    print("never here, right?")
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = F.nll_loss(output, labels)
                
                # add .detach() because of the out of memory issue. (grow computational graph)
                epoch_losses += loss.sum().detach()
                epoch_accuracy += torch.sum(preds == labels).detach().item()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                        
            epoch_losses = epoch_losses / datasets_size[phase]
            epoch_accuracy = 100. * epoch_accuracy / datasets_size[phase]
            print(f'{phase} Epoch Losses : {epoch_losses:.5f} :: Accuracy : {epoch_accuracy:.5f} :: Time : {(time.time() - epoch_begin):.4f}s')
            
            if phase == 'train':
                training_loss.append(epoch_losses)
            if phase == 'valid':
                if best_valid_accuracy < epoch_accuracy:
                    best_valid_accuracy = epoch_accuracy
                    torch.save(model.state_dict(), 'final_model.pth')
                validation_loss.append(epoch_losses)
        
        print(" about to step ")
        scheduler.step()
        print (" stepped ")
    return training_loss, validation_loss


def testing(model, dataset):
    result = []
    if torch.cuda.is_available():
            model.cuda()
    for data in dataset:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        _, preds = torch.max(output, 1)
        # testing data is sequential, so label is unnecessary thing
        result += [int(element) for element in preds.tolist()]
    return result


epochs = 20
batch_size = 64
learning_rate = 1e-4
inputDim = 1
outDim = 10

model = MobileNetV3(inputDim, outDim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

dummy_input = torch.zeros(batch_size, inputDim, 28, 28)
print(pytorch_model_summary.summary(model, dummy_input))
     
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

# Load data
valid_fraction = 0.1

train_file = pd.read_csv(Path(fileDir, qFileName['train']))
# convert dataframe of pandas into numpy array
train_file = train_file.to_numpy().astype('uint8')

boundary = int(len(train_file) * (1 - valid_fraction))

train_data = DigitDataset(train_file[:boundary], transforms, phase='training')
valid_data = DigitDataset(train_file[boundary:], transforms, phase='training')

train_data_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 0
)

valid_data_loader = DataLoader(
    valid_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 0
)

datasets = {'train' : train_data_loader, 'valid' : valid_data_loader}
datasets_size = {'train' : len(train_data_loader.dataset), 'valid' : len(valid_data_loader.dataset)}

train_loss, valid_loss = training(model, datasets, datasets_size, optimizer, scheduler, epochs=epochs)


# Draw loss graph for training
epoch_x = range(1, epochs + 1)
plt.plot(epoch_x, train_loss, 'r', epoch_x, valid_loss, 'b')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


# Load test dataset
test_file = pd.read_csv(Path(fileDir, qFileName['test']))
test_file = test_file.to_numpy().astype('uint8')

test_data = DigitDataset(test_file, transforms, phase='testing')

test_data_loader = DataLoader(
    test_data,
    batch_size = 16,
    num_workers = 3
)

result = testing(model, test_data_loader)
