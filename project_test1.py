#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
import glob
from skimage.feature import hog
import time
#sklearn


# In[2]:


"""
데이터 적재 
"""


# In[3]:


car_images = []
noncar_images = []
print('0')
for root, dirs, files in os.walk('./data/vehicles/'):
    print("1")
    print("root"+ root)
    print( dirs)
    print("="*30)
    print( files)
    print("-"*30)
    for file in files:
        print("2")
        if file.endswith(".png"):
            print('3')
            car_images.append(os.path.join(root, file))
            print('4')
            
print('no0')
for root, dirs, files in os.walk('./data/non-vehicles/'):
    print("no1")
    print("no_root"+ root)
    print( dirs)
    print("#"*30)
    print( files)
    print("*"*30)
    for file in files:
        print("no2")
        if file.endswith(".png"):
            print('no3')
            noncar_images.append(os.path.join(root, file))
            print('no4')


# In[4]:


car_images


# In[5]:


noncar_images


# In[6]:


"""
이미지 특성 추출 과 레이블링

"""


# In[7]:


#자동차 정답 특성 추출
start = time.time()
for car_image in car_images:
    img = mpimg.imread(car_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_car_feature, hog_car_image = hog(gray, orientations=9, pixels_per_cell=(4,4), 
                                 cells_per_block=(2, 2), block_norm='L2-Hys', 
                                 transform_sqrt=False, visualize = True, 
                                 feature_vector=False)
end = time.time()
print(f'{end - start:.4f}sec')


# In[20]:


car_images[1]
#파일명에 대한 리스트 


# In[21]:


hog_car_image[1]
#정답 리스트


# In[22]:


hog_car_feature[1]


# In[23]:


hog_car_feature_answer = hog_car_feature
hog_car_feature_answer


# In[ ]:





# In[ ]:





# In[8]:


start = time.time()
for noncar_image in noncar_images:
    img = cv2.imread(noncar_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_noncar_feature, hog_noncar_image = hog(gray, orientations=9, pixels_per_cell=(4, 4), 
                                cells_per_block=(2, 2), block_norm='L2-Hys',
                                transform_sqrt=False, visualize=True,
                                feature_vector=False)
end = time.time()
print(f'{end - start:.4f}sec')


# In[ ]:





# In[ ]:





# In[9]:


"""
이미지 시각화
HOG
"""


# In[10]:


car_images_test = car_images[:10]
#테스트 용 10개만 선택


# In[11]:


"""
car이미지
"""


# In[12]:


for car_image_show in car_images_test:
    img_show = mpimg.imread(car_image_show)
    gray_show = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_car_feature_show, hog_car_image_show = hog(gray_show, orientations=9, pixels_per_cell=(4,4), 
                                 cells_per_block=(2, 2), block_norm='L2-Hys', 
                                 transform_sqrt=False, visualize = True, 
                                 feature_vector=False)
    """#pixels_per_cell 픽셀 사이즈로 셀 생성
    cells_per_block - 셀 크기로 블록 생성
    orientations 방향
    """


    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img_show)

    plt.subplot(122)
    plt.imshow(hog_car_image_show, cmap='gray')


# In[13]:


"""
notcar 이미지 
"""


# In[14]:


noncar_images_test = noncar_images[:10]
#테스트 용 10개만 선택


# In[15]:


for noncar_image_show in noncar_images_test:
    img_show = cv2.imread(noncar_image_show)
    gray_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)

    hog_noncar_feature_show, hog_noncar_image_show = hog(gray_show, orientations=9, pixels_per_cell=(4, 4), 
                                cells_per_block=(2, 2), block_norm='L2-Hys',
                                transform_sqrt=False, visualize=True,
                                feature_vector=False)
    """#pixels_per_cell 픽셀 사이즈로 셀 생성
    cells_per_block - 셀 크기로 블록 생성
    orientations 방향
    """

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img_show)

    plt.subplot(122)
    plt.imshow(hog_noncar_image_show, cmap='gray')


# In[16]:


"""
레이블링 
car_images리스트 갯수 만큼 리스트 1
non_car_images 리스트 갯수 만큼 리스트 0
"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




