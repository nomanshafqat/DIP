import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.ndimage
from skimage.morphology import reconstruction

"Noman Shafqat SE5A 111572"

'''
Q1:Are any of the results satisfying?
No , none of these look satisfying 
 
Q2: Noise is gone but the corners of the tools also erode

Q3 : CH fills the whole and subtracting from the image gives us inner holes. 
THen we open the image and subtract the inner holes to obtain the noiseless image

Q4: it is identical to the result of task 3

'''

def main():
    task1()
    task2()
    task3()
    task4()


def task1():
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    image=cv2.imread("tools.png",0)

    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    img_erosion = cv2.erode(image, kernel, iterations=1)
    img_dilation = cv2.dilate(image, kernel, iterations=1)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    plt.figure("task1",(10,8))

    plt.subplot(321)
    plt.title("dilated")
    plt.imshow(img_dilation, cmap='gray')
    plt.axis('off')

    plt.subplot(322)
    plt.title("img_erosion")
    plt.imshow(img_erosion, cmap='gray')
    plt.axis('off')

    plt.subplot(323)
    plt.title("closing")
    plt.imshow(closing, cmap='gray')
    plt.axis('off')

    plt.subplot(324)
    plt.title("opening")
    plt.imshow(opening, cmap='gray')
    plt.axis('off')

    plt.subplot(325)
    plt.title("original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    #plt.show()


def task2():
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.imread("tools.png", 0)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



    close_open=cv2.morphologyEx(cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    open_close=cv2.morphologyEx(cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    plt.figure("task2",(10,8))
    plt.subplot(221)
    plt.title("close_open")
    plt.imshow(close_open, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title("open_close")
    plt.imshow(open_close, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title("original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')



    #plt.show()


def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array


def task3():
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.imread("tools.png", 0)
    ret, im_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    CH=flood_fill(im_th)
    temp=cv2.subtract(CH, im_th)
    H=cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

    TEMP=cv2.morphologyEx(CH, cv2.MORPH_OPEN, kernel)
    result=cv2.subtract(TEMP,H)

    plt.figure("task3", (10, 8))

    plt.subplot(221)
    plt.title("CH")
    plt.imshow(CH, cmap='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title("H")
    plt.imshow(H, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title("result")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title("original")
    plt.imshow(im_th, cmap='gray')
    plt.axis('off')
    #plt.show()

def task4():

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    image = cv2.imread("tools.png", 0)
    ret, im_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    background = reconstruction(opening, closing)

    plt.figure("task4", (10, 8))

    plt.subplot(221)
    plt.title("result")
    plt.imshow(background, cmap='gray')
    plt.axis('off')
    plt.show()
main()