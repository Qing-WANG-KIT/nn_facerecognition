# NN-Practical Face Recognition 


## Task introduction

![image.png](https://i.loli.net/2021/01/22/QFXWYLZi4OKuS1U.png)

##  Implementation and Application
![image-20210122171709829.png](https://i.loli.net/2021/01/23/MOfje69ZDkh5HqJ.png)

### How to use

1. requirements: pytorch>=1.7 torchvision>=0.8.1   

2. git clone https://git.scc.kit.edu/ukwet/nn_facerecognition.git

3. Unzip the （[Dataset](https://drive.google.com/file/d/1-BvF6jRnlMIJc3xjoaQWSk_mznnOLpPc/view?usp=sharing)） to the directory `./data/`

4. In the current directory terminal  :

   running `python Which_celebrity_you_look_most_similar_to.py`

   Enter the **absolute path** of an `jpg` picture. It will show which celebrity you look most similar to.

   

![image.png](https://i.loli.net/2021/01/23/kXZ9ezlm5ypKr6a.png)   
## 0 Image Augmentation：Custom transforms

### 1. Use Face Crop
+ Dataset images

  ![image.png](https://i.loli.net/2020/12/29/784gYCOVLsvBDdZ.png)   

+ Use `face_crop` transform —— `face_crop.py`

  ![image.png](https://i.loli.net/2020/12/29/8Muh1OmHig7kFYw.png)

### 2. Use Sample pairing

+ After using `face_crop` then using `Sample pairing` transform

  ![image.png](https://i.loli.net/2020/12/29/EPJMZtRIypo7ik2.png)

  Note: Sample pairing is inserted into the training intermittently to enhance the data, instead of using it in `transform` when reading the data



## 1. Data preparation

Read data：

```python
import torch
import torchvision
from torchvision import datasets,transforms 
from face_crop import face_crop


train_dir = './data/face_data/train'
valid_dir = './data/face_data/val'

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    face_crop(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std),
    transforms.RandomErasing(),
])


valid_transform = transforms.Compose([
    face_crop(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std),
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                           shuffle=True, num_workers=4)


valid_data = datasets.ImageFolder(root=valid_dir,
                                transform=valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64,
                                          shuffle=True, num_workers=4)
```



## 2.  Project model

Transfer learning based on **Backbone**: resnet50:

- [x] Model 1: Resnet50 feature extraction + 2-layer fully connected layer classifier
- [x] Model 2: Resnet50 feature extraction + 1-layer fully connected layer classifier

```python
# Resnet50 with One fully connected layers
from torchvision import models
from torch import nn
from collections import OrderedDict

model_resnet50 = models.resnet50(pretrained=True)
for param in model_resnet50.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc', nn.Linear(2048, len(class_names))),
    ('output', nn.LogSoftmax(dim=1))
]))

# Replace the classifier part of the introduced net!
model_resnet50.fc = classifier
```



## 3. Training process

Make 5 training process: check the corresponding `.pynb` file for the specific process

**Training process 1**: Resnet50 feature extraction  + 2-layer fully connected layer classifier

+ Loss function：`NLLLoss`

+ sample pairing: `False`

+ epoch： `200`



**Training process 2**: Resnet50 feature extraction + 1-layer fully connected layer classifier

+ Loss function：`NLLLoss`

+ sample pairing: `False`

+ epoch： `50`

**Training process 3**: Resnet50 feature extraction + 1-layer fully connected layer classifier

+ Loss function：`Focal loss`

+ sample pairing: `False`

+ epoch： `50`



**Training process 4**: Resnet50 feature extraction + 1-layer fully connected layer classifier

+ Loss function：``NLLLoss``

+ sample pairing: `False`

+ epoch： `100`

**Training process 5**: Resnet50 feature extraction + 1-layer fully connected layer classifier

- Loss function：``NLLLoss``

+ sample pairing: `True`

+ epoch： `100`



one example：

```python
import torch 
import torch.nn.functional as F
from torch import nn, optim  
from tqdm import tqdm
import numpy as np
from focal_loss import FocalLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model='resnet50', data_augumentation=False, loss='NLLLoss', lr=0.003, weight_decay=1e-5, epoch=None):
    if model == 'model_resnet50':
        model = model_resnet50
    
    model.to(device)
    
    if data_augumentation:
        pass
        
    if loss == 'NLLLoss':
        print('use NLLLoss')
        criterion = nn.NLLLoss()
        
    if loss == 'FocalLoss':
        print('use focal loss')
        criterion = FocalLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)    
    
    
    epochs = epoch
    train_losses, valid_losses = [], []
    valid_loss_min = np.inf

    for e in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy = 0.0

        model.train()
        for images, labels in tqdm(train_loader):
            if data_augumentation == 'samplePairing' and e+1 >= epochs * 0.2 and e+1 < epochs * 0.8:
                # do sample pairing for every image
                for i in range(images.shape[0]):
                    image_sample = SamplePairing()
                    image_a = images[i]
                    image_b = images[np.random.randint(0, images.shape[0])]
                    # images[i] is of type tensor, and both inputs are of type tensor
                    images[i] = image_sample(image_a, image_b)
                    
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss = loss.requires_grad_()
            loss.backward()

            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()  # Close dropout
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Verify loss
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                valid_loss += loss.item() * images.size(0)
                
                # Verify accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # An epoch loss 
        train_loss = train_loss/len(train_loader.sampler)   
        valid_loss = valid_loss/len(valid_loader.sampler)
        valid_accuracy = accuracy / len(valid_loader)

        # Add the loss to the list for graphing
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Add the code to save this list to the local to use this loss transformation list when not training in the future, 
        # such as comparing different models
        train_loss_array = np.array(train_losses)
        valid_loss_array = np.array(valid_losses)
        valid_accuracy_array = np.array(valid_accuracy)
        
        
        np.save('./results/Training_5/Ftp5_train_loss_array.npy', train_loss_array)
        np.save('./results/Training_5/Ftp5_valid_loss_array.npy', valid_loss_array)
        np.save('./results/Training_5/Ftp5_valid_accuracy_array.npy', valid_accuracy_array)
        
   
        # Print an epoch information
        print('Epoch {}/{}..'.format(e + 1, epochs),
              'Train loss:{:.4f}..'.format(train_loss),
              'Valid loss:{:.4f}..'.format(valid_loss),
              'Valid accuracy:{:.4f}%..'.format(valid_accuracy * 100))
        
        # Save the optimal model
        if valid_loss <= valid_loss_min:
            print('valid_loss decreased: ({:.4f} --> {:.4f}), saving model "Training_5_Res_NLLLoss_1FC_100e_SP.pt"..'.format(valid_loss_min, valid_loss))
            torch.save(model, './model/Training_5_Res_NLLLoss_1FC_100e_SP.pt')
            valid_loss_min = valid_loss
```

### Save Model

+ Save the model when the validation set loss is the smallest. Our model is saved in the path `./model`
+ The training loss and verification loss of the training process are saved in the corresponding path `./results/Training{}`

An example of train and valid loss：

![image.png](https://i.loli.net/2021/01/23/mlNwbcaIO3ZA1FU.png)



## 4. Model accuracy test
check the corresponding `Test Accuracy.pynb` file for the specific process   
all_model_test_accuracy：
1. model 1: 66.9333%
2. model 2: 69.8000%
3. model 3: 69.7167%
4. model 4: 69.8000%
5. **model 5: 70.1500%**     

![image.png](https://i.loli.net/2021/01/23/OpPI5qhQfuid3sV.png)





