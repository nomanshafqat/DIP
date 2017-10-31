import cv2
import numpy as np

def connected_components(image):

    # find the connected compoenents
    output = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)

    # return (no_of_components,labeledmatrix)
    return (output[0] - 1, output[1])


def lab7():
    image = cv2.imread("text.PNG", 0)
    image=np.invert(image)

    blackmask = cv2.inRange(image, np.array([200]), np.array([255]))
    #   cv2.imshow("blackmast",blackmask)

    ret, thresh = cv2.threshold(blackmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imshow("thresh",thresh)

    # find connected components
    no, matrix = connected_components(thresh)


    for a in matrix:
        for b in a:
            print(str(b)+"\t",end="")
        print("")

    for i in range(1, no):
        temp = np.where(matrix == i)
        a = np.min(temp[0])
        c = np.max(temp[0])
        b = np.min(temp[1])
        d = np.max(temp[1])

        roi = thresh[a:c, b:d]

        if (len(roi) > 0):
            cv2.imwrite("problem 10-img" + str(i) + ".png", np.invert(roi))

lab7()