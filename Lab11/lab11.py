import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import exposure
import scipy.ndimage
import mahotas
from skimage.morphology import skeletonize, thin

"Noman Shafqat SE5A 111572"


def main():
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    plt.show()


def lab101(img):
    kernel2 = morphology.disk(2)
    kernel4 = morphology.disk(4)
    kernel6 = morphology.disk(6)
    kernel8 = morphology.disk(8)

    tophat2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)
    tophat4 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel4)
    tophat6 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel6)
    tophat8 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel8)
    thresh = 10
    tophat2[tophat2 < thresh] = 0
    tophat4[tophat4 < thresh] = 0
    tophat6[tophat6 < thresh] = 0
    tophat8[tophat8 < thresh] = 0

    tophat = cv2.multiply(cv2.add(cv2.add(tophat2, tophat4), cv2.add(tophat8, tophat6)), 2)

    return tophat


def task1():
    for i in range(1, 6):
        name = "0" + str(i) + "_test.tif"
        img = cv2.imread(name, 0)
        img = np.invert(img)

        tophat = lab101(img)

        b = np.unpackbits(tophat, axis=1)
        print(b.shape)
        shape = tophat.shape
        a = np.array([[2], [7], [23]], dtype=np.uint8)
        c = b.reshape(584, -1, 8)
        bitplane = c[:, :, 0] * 255
        bitplane1 = c[:, :, 1] * 255

        tophat = cv2.bitwise_or(bitplane, bitplane1)

        plt.figure("task1-" + name)
        plt.subplot(1, 1, 1)
        plt.title(name)
        plt.imshow(tophat, cmap='gray')
        plt.axis('off')
        break

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def task2():
    """There is more noise iun the image"""
    for i in range(1, 6):
        name = "0" + str(i) + "_test.tif"
        img = cv2.imread(name, 0)
        img = np.invert(img)

        tophat = lab101(img)

        # Adaptive Equalization
        tophat = exposure.equalize_adapthist(tophat, clip_limit=0.05)
        print(tophat.shape)
        print(tophat)

        b = np.unpackbits(np.array(np.multiply(tophat, 255)).astype("uint8"), axis=1)
        print(b.shape)
        shape = tophat.shape
        c = b.reshape(584, -1, 8)
        bitplane = c[:, :, 0] * 255
        bitplane1 = c[:, :, 1] * 255
        tophat = cv2.bitwise_or(bitplane, bitplane1)

        plt.figure("task2-" + name)
        plt.subplot(1, 1, 1)
        plt.title(name)
        plt.imshow(tophat, cmap='gray')
        plt.axis('off')

        break


def task3():
    '''
    q#1:
    Boxex
    q#2 : rotate the image
    Q#3:
     SE2 = [
            0, 0, 0, 0, 0,
            -1, -1, -1, -1, 0,
            0, 1, 1, -1, 0,
            0, 0, 1, -1, 0,
            0, 0, 0, -1, 0
        ]

    No, It won't because you cannot put 1 and -1 at one index.

    '''
    name = "blobs.png"
    img = cv2.imread(name, 0)
    img = np.invert(img)
    SE1 = [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0
    ]
    SE2 = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0
    ]

    SE1 = np.array(SE1).reshape(5, 5)
    SE2 = np.array(SE2).reshape(5, 5)

    img = np.array(img)
    hit = scipy.ndimage.morphology.binary_hit_or_miss(img, structure1=SE1, structure2=SE2).astype(np.int)
    hit = cv2.multiply(hit, 255)

    print(np.count_nonzero(np.array(hit)))

    print(hit)
    plt.figure("task3-" + name)
    plt.subplot(1, 1, 1)
    plt.title(name)
    plt.imshow(hit, cmap='gray')
    plt.axis('off')


def task4():

    '''Not exaclty but somewhat similar'''

    name = "blobs.png"
    image = cv2.imread(name, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Dilated = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)

    result = np.subtract(Dilated, image)

    bwperim = mahotas.labeled.bwperim(image, n=8)

    plt.figure("task4-" + name)
    plt.subplot(1, 2, 1)
    plt.title(name + "Dilation")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(name + "bwperim")
    plt.imshow(bwperim, cmap='gray')
    plt.axis('off')


def task5():
    '''It gets more thinner but at logarithmic rate'''
    name = "blobs.png"
    image = cv2.imread(name, 0)

    ret, im_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    im_th[im_th > 1] = 1

    print(im_th)
    skeleton = thin(im_th)

    plt.figure("task5-" + name)
    plt.subplot(1, 2, 1)
    plt.title(name + "thin")
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')


def task6():
    '''Results are relatively similar'''
    name = "blobs.png"
    image = cv2.imread(name, 0)

    ret, im_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    im_th[im_th > 1] = 1

    print(im_th)
    skeleton = skeletonize(im_th)

    plt.figure("task6-" + name)
    plt.subplot(1, 2, 1)
    plt.title(name + "Skeltopn")
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')


main()
