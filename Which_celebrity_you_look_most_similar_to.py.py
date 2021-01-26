# -*- coding: utf-8 -*-
"""
Introduction：
+ Test Model：Training_5_Res_NLLLoss_1FC_100e_SP.pt
+ Model Path：'./model/Training_5_Res_NLLLoss_1FC_100e_SP.pt'
+ Catalog of Test Imges：./data/face_data_test  【include group members】

## Test Image_1 `"001name.jpg"` ：Visualization
+ Test Image
"""

"""## Application
+ Load Model
+ Load Picture
+ Forecast Result
+ Visualization of Results
"""
import torch
import numpy as np
from torch import Tensor
import os
from MTCNN.detector import detect_faces
from MTCNN.visualization_utils import show_results
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from face_crop import face_crop
# Commented out IPython magic to ensure Python compatibility.
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_path = input("Please enter the absolute path of the picture(only jpg):\n")
model_path = "./model/Training_5_Res_NLLLoss_1FC_100e_SP.pt"
out_label = "".join(list(img_path)[38:len(list(img_path)) - 4])


"""Function to load Images"""
def load(img_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
        face_crop(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    img = Image.open(img_path)
    img = img_transform(img)

    ###############################
    # look over transform
    # image = trans(img)
    # print(type(image))
    # display(image)
    ###############################

    img = img.unsqueeze(0)  # torch.Size([3, 2541, 1920]) -> torch.Size([1, 3, 2541, 1920])
    img = img.to(device)
    return img

def Visualization(choise, img_path, model_path):
    model = torch.load(model_path).to(device)
    with torch.no_grad():
        log_ps = model.forward(load(img_path))
        outputs = torch.exp(log_ps)
    if choise == "chart":
        show_chart(outputs)

def show_chart(outputs):
    top_p, top_k = torch.topk(outputs, 10, dim=1, largest=True, sorted=True, out=None)
    img = Image.open(img_path)
    bbox, landmarks = detect_faces(img)
    img = show_results(img, bbox)
    draw = ImageDraw.Draw(img)
    class_names = np.load("./class_names.npy")
    name = class_names[top_k.squeeze()[0].item()]
    text = 'Sim:{:.1f}%\nLab:{}'.format(top_p.squeeze()[0].item()*100, name)
    ###
    fontsize = 1  # starting font size

    # portion of image width you want text width to be
    img_fraction = 0.6

    font = ImageFont.truetype("./font/ebrima.ttf", fontsize)
    while font.getsize(text)[0] < img_fraction*img.size[1]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("./font/ebrima.ttf", fontsize)

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype("./font/ebrima.ttf", fontsize)

    ###
    
    draw.text((bbox[0][0], bbox[0][1]), text,  fill ="white", font=font)

    top_p = Tensor.cpu(top_p).numpy()

    top_p = top_p[0].tolist()
    class_names = np.load("./class_names.npy")
    heng = [[] for j in range(10)]
    for i in range(10):
        label_top10 = class_names[top_k.squeeze()[i].item()]
        heng[i] = label_top10

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    ori_img = Image.open(img_path)
    plt.title("Recognized Image \n {}".format(out_label), fontsize="x-large")
    plt.imshow(img)
    plt.subplot(3, 3, 3)
    top_p = list(reversed(top_p))
    heng = list(reversed(heng))
    plt.barh(range(len(top_p)), top_p, tick_label=heng)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.title("Top10 Propotion of Similarity", fontsize="xx-large")

    top_p = list(reversed(top_p))
    heng = list(reversed(heng))
    plt.subplot(3, 3, 5)

    label_first = class_names[top_k.squeeze()[0].item()]
    pic_first = str(label_first) + "_0001" + ".jpg"
    ref_imgpath = os.path.join("./data/lfw", str(label_first), pic_first)
    ref_imgpath = ref_imgpath.replace('\\', '/')
    first_img = Image.open(ref_imgpath)
    plt.title("Top1 Similarity:{}% \n {}".format('%.3f' % (top_p[0] * 100), label_first), fontsize="x-large")
    plt.imshow(first_img)

    for i in range(1, 4):
        label_s = class_names[top_k.squeeze()[i].item()]
        pic = str(label_s) + "_0001" + ".jpg"
        ref_imgpath = os.path.join("./data/lfw", str(label_s), pic)
        ref_img = Image.open(ref_imgpath)
    
        plt.subplot(3, 3, i + 6)
        plt.title("Top{} Similarity:{}% \n {}".format(i + 1, '%.3f' % (top_p[i] * 100), label_s), fontsize="x-large", y=-0.3)
        plt.imshow(ref_img)

    plt.show()


def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read picture name
    # The picture does not need to be placed in the specified path, use the absolute path

    model = torch.load(model_path).to(device)

    with torch.no_grad():
        log_ps = model.forward(load(img_path))
        outputs = torch.exp(log_ps)
    print(img_path)
    show_chart(outputs)


if __name__ == "__main__":
    print("Which celebrity you look most similar to?")
    print("...")
    main()
