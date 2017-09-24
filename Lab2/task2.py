from PIL import Image
import numpy as np
import cv2


image = cv2.imread('lab2-task2-image.jpg')
#print(image)

orig_arr = np.array(image)

#print(orig_arr)
dict={}
for i,a in enumerate(orig_arr):
        for j,pixel in enumerate(a):
            if pixel[0] in range(20,55) and pixel[1] in range(15,45) and pixel[2] in range(200,255):
                orig_arr[i][j]=[0 ,255 ,255]
            elif pixel[0] in range(70, 80) and pixel[1] in range(150, 255) and pixel[2] in range(30, 40):
                orig_arr[i][j] = [255, 255, 0]
            elif pixel[0] in range(204,210) and pixel[1] in range(65, 75) and pixel[2] in range(55, 65):
                orig_arr[i][j] = [255, 0, 255]

                #print("found")
            #print (pixel,end="\n")



cv2.imwrite('messigray.png', orig_arr)
