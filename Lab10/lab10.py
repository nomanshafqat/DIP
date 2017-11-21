import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology

"Noman Shafqat SE5A 111572"

def main():
    task1()
    task2()
    plt.show()


def task1():
    plt.figure("task1")

    for i in range(1, 6):

        name = "0" + str(i) + "_test.tif"
        img = cv2.imread(name, 0)
        img = np.invert(img)
        print(img)

        kernel0 = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ]).reshape(5, 5)

        kernel45 = np.array([
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1
        ]).reshape(5, 5)
        print(kernel0)
        kernel90 = np.rot90(kernel0)
        kernel135 = np.rot90(kernel45)

        print(kernel90)
        tophat0 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel0)
        tophat45 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel45)
        tophat90 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel90)
        tophat135 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel135)

        thresh = 3

        tophat0[tophat0 <= thresh] = 0
        tophat45[tophat45 <= thresh] = 0
        tophat90[tophat90 <= thresh] = 0
        tophat135[tophat135 <= thresh] = 0

        tophat = cv2.multiply(cv2.add(cv2.add(tophat0, tophat90), cv2.add(tophat45, tophat135)), 4)
        # tophat[tophat>thresh]=255
        # tophat[tophat<=thresh]=0

        tophat = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        #cv2.imshow("sadas" + str(i), np.invert(tophat))
        #cv2.imshow("orig" + str(i), img)

        plt.subplot(5, 1, i)
        plt.title(name)
        plt.imshow(np.invert(tophat), cmap='gray')
        plt.axis('off')

        # cv2.waitKey(0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def task2():
    plt.figure("task2")

    for i in range(1, 6):
        name="0" + str(i) + "_test.tif"
        img = cv2.imread(name, 0)
        img = np.invert(img)

        kernel2 = morphology.disk(2)
        kernel4 = morphology.disk(4)
        kernel6 = morphology.disk(6)
        kernel8 = morphology.disk(8)


        tophat2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)
        tophat4 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel4)
        tophat6 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel6)
        tophat8 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel8)
        thresh=10
        tophat2[tophat2<thresh]=0
        tophat4[tophat4<thresh]=0
        tophat6[tophat6<thresh]=0
        tophat8[tophat8<thresh]=0


        tophat = cv2.multiply(cv2.add(cv2.add(tophat2, tophat4), cv2.add(tophat8, tophat6)), 4)
        # tophat[tophat>thresh]=255
        # tophat[tophat<=thresh]=0

        # tophat = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        #cv2.imshow("sadas" + str(i), np.invert(tophat))
        #cv2.imshow("orig" + str(i), img)
        plt.subplot(5,1,i)
        plt.title(name)
        plt.imshow(np.invert(tophat), cmap='gray')
        plt.axis('off')

        #cv2.waitKey(0)
    #cv2.destroyAllWindows()


main()
