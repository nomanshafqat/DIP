import cv2
from skimage import morphology
import numpy as np
import scipy as sp
import scipy.ndimage
from skimage import io, color


"Noman Shafqat SE5A 111572"

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

def connected_components(image):

    # find the connected compoenents
    output = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)

    # return (no_of_components,labeledmatrix)
    return (output[0] - 1, output[1])


refPt = []
clicked=0
labImg=0
def click_and_crop(event, x, y, flags, param):
    img = cv2.imread("mnms.jpg")

    # grab references to the global variables
    global refPt,labImg
    print(refPt)
    if event == cv2.EVENT_LBUTTONUP:
        refPt=[y,x]
        clicked=1
        acom = labImg[:, :, 1]
        bcom = labImg[:, :, 2]

        aref=acom[refPt[0]][refPt[1]]
        bref=bcom[refPt[0]][refPt[1]]
        print(aref,bref)
        temp=np.array(labImg)
        distance=np.zeros((temp.shape[0],temp.shape[1]))

        for i in range (temp.shape[0]):
            for j in range(temp.shape[1]):


                distance[i][j]=np.sqrt(np.add(np.power(np.subtract(labImg[i][j][1],aref),2),np.power(np.subtract(labImg[i][j][2],bref),2)))
                #print(distance[i][j])
        #print(labImg)
        '''
        for i in range (temp.shape[0]):
            for j in range(temp.shape[1]):
                #print(labImg[i][j])
                continue

        '''
        for i in range (temp.shape[0]):
            for j in range(temp.shape[1]):
                if (distance[i][j]>10):
                    img[i][j][0]=255
                    img[i][j][1]=255
                    img[i][j][2]=255

        print(distance.shape)

        #cv2.imshow("disytance", distance)
        cv2.imshow("result", img)






def main():
    img = cv2.imread("mnms.jpg")

    # TAsk 1 Thresholding
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    bbw = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gbw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rbw = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bbw=np.invert(bbw)
    gbw=np.invert(gbw)
    rbw=np.invert(rbw)

    combined = cv2.bitwise_or(bbw, cv2.bitwise_or(gbw, rbw))
    #cv2.imshow("sd", combined)


    # Task    2.   Morphological   processing

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    openedImage=morphology.opening(combined,kernel)
    filled=flood_fill(openedImage)


    #cv2.imshow("opened", openedImage)
    #cv2.imshow("filled", filled)

    # Labeling connected components
    connectedImg=connected_components(filled)
    conn=connectedImg[1]
    for i in range (0,connectedImg[0]+1):
        temp=np.where(conn==i)
        indexr=r[temp]
        indexg=g[temp]
        indexb=b[temp]

        medianr=np.median(indexr)
        mediang=np.median(indexg)
        medianb=np.median(indexb)
        for i in range(0,len(temp[0])):
            img[temp[0][i]][temp[1][i]][0]=medianb
            img[temp[0][i]][temp[1][i]][1]=mediang
            img[temp[0][i]][temp[1][i]][2]=medianr

        global labImg
        labImg = color.rgb2lab(img)

        cv2.imshow("preseg", img)
        cv2.setMouseCallback("preseg", click_and_crop)
        global clicked
        #while clicked==0:
        #    time.sleep(1)
        #    print(".",end="")
        #    continue





    cv2.waitKey()
    cv2.destroyAllWindows()


main()
