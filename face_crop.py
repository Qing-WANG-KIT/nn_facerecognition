#!/usr/bin/env python
# coding: utf-8

# In[3]:


from PIL import Image
from MTCNN import detector
from MTCNN.detector import detect_faces

class face_crop(object):
    '''
    自定义transform
    '''
    def __init__(self, mode=True):# ...是要传入的多个参数
        # 对多参数进行传入
        # 如 self.p = p 传入概率
        # ...
        super().__init__()
 
    def __call__(self, img): # __call__函数还是只有一个参数传入
        # 该自定义transforms方法的具体实现过程
        # 实现图片的面部识别与crop
        self.bounding_boxes, self.landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
        self.bounding_boxes =self.bounding_boxes[0][:4]
        x1 = self.bounding_boxes[0]-10
        y1 = self.bounding_boxes[1]-10
        x2 = self.bounding_boxes[2]+10
        y2 = self.bounding_boxes[3]+10
        self.bounding_boxes = (x1, y1, x2, y2)
        img = img.crop(self.bounding_boxes)
        return img 

