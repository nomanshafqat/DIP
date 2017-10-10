import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage


def task1():
    cartoon = cv2.imread("cartoon.tif", 0)

    kernel = [
        0, -1, 0,
        -1, 1 + 4, -1,
        0, -1, 0
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(421)
    plt.imshow(dst808)
    plt.title("1+4")
    plt.axis('off')

    kernel = [
        0, -1, 0,
        -1, 2 + 4, -1,
        0, -1, 0
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(423)
    plt.imshow(dst808)
    plt.title("2+4")
    plt.axis('off')

    kernel = [
        0, -1, 0,
        -1, 3 + 4, -1,
        0, -1, 0
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(425)
    plt.imshow(dst808)
    plt.title("3+4")
    plt.axis('off')

    kernel = [
        -1, -1, -1,
        -1, 1 + 8, -1,
        -1, -1, -1
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(422)
    plt.imshow(dst808)
    plt.title("1+8")
    plt.axis('off')

    kernel = [
        -1, -1, -1,
        -1, 2+ 8, -1,
        -1, -1, -1
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(424)
    plt.imshow(dst808)
    plt.title("2+8")
    plt.axis('off')

    kernel = [
        -1, -1, -1,
        -1, 4 + 8, -1,
        -1, -1, -1
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    dst808 = cv2.filter2D(cartoon, -1, kernel)

    plt.subplot(426)
    plt.imshow(dst808)
    plt.title("4+8")
    plt.axis('off')

    newimg=cv2.multiply(cartoon,3)

    plt.subplot(427)
    plt.imshow(newimg)
    plt.title("multiply*3")
    plt.axis('off')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task3():

    image = cv2.imread("pears.png", 0)

    blurr = cv2.GaussianBlur(image, (11, 11), 1)
    image = cv2.add(image, cv2.subtract(image, blurr))
    plt.subplot(421)
    plt.imshow(blurr, cmap='gray')
    plt.axis('off')

    plt.subplot(422)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    blurr = cv2.GaussianBlur(image, (15, 15), 10.0)
    image = cv2.add(image, cv2.subtract(image, blurr))


    plt.subplot(423)
    plt.imshow(blurr, cmap='gray')
    plt.axis('off')

    plt.subplot(424)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    blurr = cv2.GaussianBlur(image, (25, 25), 10.0)
    image = cv2.add(image, cv2.subtract(image, blurr))

    plt.subplot(425)
    plt.imshow(blurr, cmap='gray')
    plt.axis('off')

    plt.subplot(426)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    blurr = cv2.GaussianBlur(image, (35, 35), 10.0)
    image = cv2.add(image, cv2.subtract(image, blurr))

    plt.subplot(427)
    plt.imshow(blurr, cmap='gray')
    plt.axis('off')

    plt.subplot(428)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def gradient(img,flag):
    if(flag==0):
        cartoon2 = img.copy()
        cartoon = np.pad(img, ((0, 0), (1, 0)), mode='constant')[:, :-1]
        print(cartoon)

        result = cv2.subtract(cartoon2, cartoon)
    else:
        cartoon2 = img.copy()
        cartoon = np.pad(img, ((1, 0), (0, 0)), mode='constant')[:-1,:]
        print(cartoon.shape)
        print(cartoon2.shape)

        print(cartoon)

        result = cv2.subtract(cartoon, cartoon2)
    return result

def task4():
    cartoon = cv2.imread("pears.png", 0)


    vertical=gradient(cartoon,1)
    hor=gradient(cartoon,0)

    #cv2.imshow("Vertical",vertical)
    #cv2.imshow("Horizontal",hor)
    plt.subplot(211)
    plt.imshow(vertical, cmap='gray')
    plt.axis('off')
    plt.title("vertical")

    plt.subplot(212)
    plt.imshow(hor, cmap='gray')
    plt.axis('off')
    plt.title("Horizontal")

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task2():
    cartoon = cv2.imread("cartoon.tif", 0)

    str=scipy.ndimage.filters.gaussian_laplace(cartoon,2.0,mode="nearest").astype("uint8")

    print(str.shape)
    print(cartoon.shape)

    plt.subplot(211)
    plt.imshow(str, cmap='gray')
    plt.axis('off')
    plt.title("guassian-alplacian")

    plt.subplot(212)
    plt.axis('off')
    plt.title("Horizontal")

    add=np.hypot(cartoon.astype(float),str.astype(float))

    plt.imshow(add, cmap='gray')

    plt.show()

    #cv2.imshow("asdas",str)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#task1()
#task3()
task2()
#task4()