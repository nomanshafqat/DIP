import cv2
import numpy as np
import scipy
from   skimage.filters import roberts, sobel, prewitt
from scipy import ndimage
from matplotlib import pyplot as plt

'''
Noman Shafqat
'''

'''
weighted average filters give different weights to different neighbouring pixels but in our image there is no clear difference;

NOTE : TASK1 AND TASK 2 FUNCTIONS AT THE END OF THE FILE COMMENT ONE OUT TO RUN
'''


def task1():
    img = cv2.imread('pears.png',0)
    kernel = np.ones((3, 3), np.float32) / 9

    dst3 = cv2.filter2D(img, -1, kernel)

    kernel = np.ones((5, 5), np.float32) / 25
    dst5 = cv2.filter2D(img, -1, kernel)

    kernel = np.ones((15, 15), np.float32) / 225
    dst15 = cv2.filter2D(img, -1, kernel)

    kernel = np.ones((35, 35), np.float32) / 1225
    dst35 = cv2.filter2D(img, -1, kernel)

    kernel = [
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    kernel = kernel / 16
    dst242 = cv2.filter2D(img, -1, kernel)

    kernel = [
        4, 8, 4,
        8, 0, 8,
        4, 8, 4
    ]
    kernel = np.array(kernel)
    kernel = kernel.reshape(3, 3)
    kernel = kernel / 48
    dst808 = cv2.filter2D(img, -1, kernel)

    #print(kernel)
    #cv2.imshow("Original", img)
    # cv2.imshow("3*3 average",dst3)
    # cv2.imshow("5*5 average",dst5)
    # cv2.imshow("15*15 average",dst15)
    # cv2.imshow("35*35 average",dst35)
    # cv2.imshow("3*3 242 average",dst242)
    #cv2.imshow("3*3 848 average", dst808)

    plt.subplot(331)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(332)
    plt.title('dst3')
    plt.imshow(dst3.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(333)
    plt.title('dst5')
    plt.imshow(dst5.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(334)
    plt.title('dst15')
    plt.imshow(dst15.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(335)
    plt.title('dst35')
    plt.imshow(dst35.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(336)
    plt.title('dst242')
    plt.imshow(dst242.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(337)
    plt.title('dst808')
    plt.imshow(dst808.astype('uint8'), cmap='gray')

    plt.axis('off')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def task2():
    img = cv2.imread('pears.png', 0)

    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    dx = ndimage.sobel(img.astype("float32"), 0)
    dy = ndimage.sobel(img.astype("float32"), 1)  # vertical derivative
    sobeled = np.hypot(dx, dy)  # magnitude

    vprevit = [
        1, 1, 1,
        0, 0, 0,
        -1, -1, -1
    ]
    hprevit = [
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    ]
    vprevit = np.array(vprevit)
    vprevit = vprevit.reshape(3, 3)
    dy = cv2.filter2D(img, -1, vprevit)

    hprevit = np.array(hprevit)
    hprevit = hprevit.reshape(3, 3)
    dx = cv2.filter2D(img, -1, hprevit)
    previt = np.hypot(dx, dy)  # magnitude

    laplacianker = [-1, -1, -1,
                    -1, 8, -1,
                    -1, -1, -1]

    laplacianker = np.array(laplacianker)
    laplacianker = laplacianker.reshape(3, 3)
    dx = cv2.filter2D(img, -1, laplacianker)
    laplacian = np.hypot(dx, dy)

    robert1 = [0, 1,
               -1, 0]
    robert2 = [-1, 0,
               0, 1]

    robert2 = np.array(robert2)
    robert2 = robert2.reshape(2, 2)
    dy = cv2.filter2D(img, -1, robert2)

    robert1 = np.array(robert1)
    robert1 = robert1.reshape(2, 2)
    dx = cv2.filter2D(img, -1, robert1)
    robert = np.hypot(dx, dy)

    kirch1 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    kirch2 = [-2, -1, 0, -1, 0, 1, 0, 1, 2]
    kirch3 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    kirch4 = [0, -1, -2, 1, 0, -1, 2, 1, 0]

    kirch1 = np.array(kirch1).reshape(3, 3)
    kirch2 = np.array(kirch2).reshape(3, 3)
    kirch3 = np.array(kirch3).reshape(3, 3)
    kirch4 = np.array(kirch4).reshape(3, 3)

    k1 = cv2.filter2D(img, -1, kirch1)
    k2 = cv2.filter2D(img, -1, kirch2)
    k3 = cv2.filter2D(img, -1, kirch3)
    k4 = cv2.filter2D(img, -1, kirch4)

    temp1 = cv2.max(k1, k2)
    temp2 = cv2.max(k3, k4)

    kirch = cv2.max(temp1, temp2)
    '''
    cv2.imshow("Original", img)
    cv2.imshow("Sobeled", sobeled.astype("uint8"))
    cv2.imshow("Prewit", previt.astype("uint8"))
    cv2.imshow("Laplacian", laplacian.astype("uint8"))
    cv2.imshow("robert", robert.astype("uint8"))
    cv2.imshow("kirch", kirch.astype("uint8"))
    '''
    plt.subplot(521)
    plt.title('Original')
    plt.imshow(img.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(322)
    plt.title('sobeled')
    plt.imshow(sobeled.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(323)
    plt.title('previt')
    plt.imshow(previt.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(324)
    plt.title('laplacian')
    plt.imshow(laplacian.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(325)
    plt.title('robert')
    plt.imshow(robert.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.subplot(326)
    plt.title('kirch')
    plt.imshow(kirch.astype('uint8'), cmap='gray')
    plt.axis('off')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


task1()
#task2()
