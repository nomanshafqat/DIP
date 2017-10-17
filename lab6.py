import skimage
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
import scipy


'''
Noman Shafqat 111572 SE5A

Question 1 : What is the effect of using the wrong sign when filtering with the contra harmonic mean filter?

When R<0  Salt noise will be removed.
When R>0  Pepper noise will be removed.

Question 2: Why do you think the median filter works on salt and pepper noise but not Gaussian noise?

Salt noise is basically pixel values nearing 255 and Pepper noise are near to 0. When median filter is applied the pixel value at the middle has more chances of being the median but on the other hand gaussian noise is random value can be any number between 0 to 255 so it’s hard to predict the effectiveness of median filter on gaussian noise.

'''

def arithmatic_mean(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):
    '''Arithmatic Mean Filter'''

    plt.figure("Arithamatic",figsize=(15,8))

    #mean kernel being made
    kernel = np.array([ 1, 1, 1,    1, 1 , 1,    1, 1, 1]).reshape(3, 3)/9

    ''' Filter being applied to each image and the result being plot into a subplot'''
    arithmaticmean = cv2.filter2D(guassiannoise, -1, kernel)
    plt.subplot(321)
    plt.imshow(arithmaticmean,cmap='gray')
    plt.title("arithmaticmean on guassiannoise")
    plt.axis('off')

    poissonnoise = cv2.filter2D(poissonnoise, -1, kernel)
    plt.subplot(322)
    plt.imshow(poissonnoise,cmap='gray')
    plt.title("arithmaticmean on poissonnoise")
    plt.axis('off')

    saltnoise = cv2.filter2D(saltnoise, -1, kernel)
    plt.subplot(323)
    plt.imshow(saltnoise,cmap='gray')
    plt.title("arithmaticmean on saltnoise")
    plt.axis('off')

    peppernoise = cv2.filter2D(peppernoise, -1, kernel)
    plt.subplot(324)
    plt.imshow(peppernoise,cmap='gray')
    plt.title("arithmaticmean on peppernoise")
    plt.axis('off')

    specklenoise = cv2.filter2D(specklenoise, -1, kernel)
    plt.subplot(325)
    plt.imshow(specklenoise,cmap='gray')
    plt.title("arithmaticmean on specklenoise")
    plt.axis('off')

    snpnoise = cv2.filter2D(snpnoise, -1, kernel)
    plt.subplot(326)
    plt.imshow(snpnoise,cmap='gray')
    plt.title("arithmaticmean on salt n pepper noise")
    plt.axis('off')

    plt.show()


def median_fileter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):


    plt.figure("Median",figsize=(15,8))

    '''Median Filter being applied to each image and the result being plot into a subplot'''

    plt.subplot(321)
    plt.imshow(filters.median(guassiannoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(filters.median(poissonnoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(filters.median(saltnoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(filters.median(peppernoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(filters.median(specklenoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(filters.median(snpnoise, np.ones((3, 3))),cmap='gray')
    plt.title("Median on salt n pepper noise")
    plt.axis('off')



    plt.show()

def max_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):
    plt.figure("Max Filter",figsize=(15,8))

    '''Max Filter being applied to each image and the result being plot into a subplot'''

    plt.subplot(321)
    plt.imshow(scipy.ndimage.filters.maximum_filter(guassiannoise, 3) ,cmap='gray')
    plt.title("Max on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(scipy.ndimage.filters.maximum_filter(poissonnoise,3),cmap='gray')
    plt.title("Max on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(scipy.ndimage.filters.maximum_filter(saltnoise,3),cmap='gray')
    plt.title("Max on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(scipy.ndimage.filters.maximum_filter(peppernoise, 3),cmap='gray')
    plt.title("Max on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(scipy.ndimage.filters.maximum_filter(specklenoise, 3),cmap='gray')
    plt.title("Max on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(scipy.ndimage.filters.maximum_filter(snpnoise, 3),cmap='gray')
    plt.title("Max on salt n pepper noise")
    plt.axis('off')



    plt.show()

def min_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):
    '''Min Filter being applied to each image and the result being plot into a subplot'''

    plt.figure("Min Filter",figsize=(15,8))

    plt.subplot(321)
    plt.imshow(scipy.ndimage.filters.minimum_filter(guassiannoise, 3) ,cmap='gray')
    plt.title("Min on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(scipy.ndimage.filters.minimum_filter(poissonnoise,3),cmap='gray')
    plt.title("Min on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(scipy.ndimage.filters.minimum_filter(saltnoise,3),cmap='gray')
    plt.title("Min on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(scipy.ndimage.filters.minimum_filter(peppernoise, 3),cmap='gray')
    plt.title("Min on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(scipy.ndimage.filters.minimum_filter(specklenoise, 3),cmap='gray')
    plt.title("Min on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(scipy.ndimage.filters.minimum_filter(snpnoise, 3),cmap='gray')
    plt.title("Min on salt n pepper noise")
    plt.axis('off')

    plt.show()

def geometric_filter(img):
    ''' Geometric mean being calculated and updated into a copy of the original image'''
    img=img.astype(float)
    w,l=img.shape
    result=img.copy()
    result=img.astype("uint8")
    for x in range(1,w-1):
        for y in range(1,l-1):
            kernel=img[x-1:x+2,y-1:y+2]
            kernel=kernel.flatten()
            prod=1
            for a in kernel:
                prod=prod*(a+0.000001)
            prod=np.power(prod,float(1/9))
            result[x][y]=prod*255
    return result

def geo_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):

    '''Geometric Mean Filter being applied to each image and the result being plot into a subplot'''

    plt.figure("geometric Filter",figsize=(15,8))

    plt.subplot(321)
    plt.imshow(geometric_filter(guassiannoise) ,cmap='gray')
    plt.title("geometric on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(geometric_filter(poissonnoise),cmap='gray')
    plt.title("geometric on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(geometric_filter(saltnoise),cmap='gray')
    plt.title("geometric on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(geometric_filter(peppernoise),cmap='gray')
    plt.title("geometric on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(geometric_filter(specklenoise),cmap='gray')
    plt.title("geometric on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(geometric_filter(snpnoise),cmap='gray')
    plt.title("geometric on salt n pepper noise")
    plt.axis('off')



    plt.show()

def contra_harmonic(img,R=-.5):
    '''Contra Harmonic filter being applied to an image and updated in a copy of image'''
    img=img.astype(float)
    w,l=img.shape
    img2=img.copy()
    img2=img.astype("uint8")
    for x in range(1,w-1):
        for y in range(1,l-1):
            kernel=img[x-1:x+2,y-1:y+2]
            kernel=kernel.flatten()
            num=np.sum(np.power(kernel,R+1))
            den=np.sum(np.power(kernel,R))
            img2[x][y]=(num/den)*255

    return img2

def contra_harmonic_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):

    ''' Contra Harmonic Filter being applied to each image and the result being plot into a subplot'''

    plt.figure("contra_harmonic Filter",figsize=(15,8))

    plt.subplot(321)
    plt.imshow(contra_harmonic(guassiannoise) ,cmap='gray')
    plt.title("contra_harmonic on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(contra_harmonic(poissonnoise),cmap='gray')
    plt.title("contra_harmonic on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(contra_harmonic(saltnoise),cmap='gray')
    plt.title("contra_harmonic on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(contra_harmonic(peppernoise),cmap='gray')
    plt.title("contra_harmonic on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(contra_harmonic(specklenoise),cmap='gray')
    plt.title("contra_harmonic on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(contra_harmonic(snpnoise),cmap='gray')
    plt.title("contra_harmonic on salt n pepper noise")
    plt.axis('off')



    plt.show()

def harmonic(img):
    '''Harmonic filter being applied and updated'''
    img=img.astype(float)
    w,l=img.shape
    img2=img.copy()
    img2=img.astype("uint8")
    for x in range(1,w-1):
        for y in range(1,l-1):
            kernel=img[x-1:x+2,y-1:y+2]
            kernel=kernel.flatten()
            prod=1
            for a in kernel:
                prod+=(1/(a+0.000001))
            prod=9/prod
            img2[x][y]=prod*255
    return img2

def harmonic_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise):

    '''Harmonic Mean Filter being applied to each image and the result being plot into a subplot'''

    plt.figure("harmonic Filter", figsize=(15, 8))

    plt.subplot(321)
    plt.imshow(harmonic(guassiannoise), cmap='gray')
    plt.title("harmonic on guassiannoise")
    plt.axis('off')

    plt.subplot(322)
    plt.imshow(harmonic(poissonnoise), cmap='gray')
    plt.title("harmonic on poissonnoise")
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(harmonic(saltnoise), cmap='gray')
    plt.title("harmonic on saltnoise")
    plt.axis('off')

    plt.subplot(324)
    plt.imshow(harmonic(peppernoise), cmap='gray')
    plt.title("harmonic on peppernoise")
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(harmonic(specklenoise), cmap='gray')
    plt.title("harmonic on specklenoise")
    plt.axis('off')

    plt.subplot(326)
    plt.imshow(harmonic(snpnoise), cmap='gray')
    plt.title("harmonic on salt n pepper noise")
    plt.axis('off')

    plt.show()


img=cv2.imread("cameraman.bmp",0)

'''Different Noise types being added to image'''
guassiannoise=skimage.util.random_noise(img,"gaussian")
poissonnoise=skimage.util.random_noise(img,"poisson")
saltnoise=skimage.util.random_noise(img,"salt")
peppernoise=skimage.util.random_noise(img,"pepper")
specklenoise=skimage.util.random_noise(img,"speckle")
snpnoise=skimage.util.random_noise(img,"s&p")

'''Each type of Noise filter being applied to the images with added noise above'''
arithmatic_mean(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
median_fileter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
max_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
min_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
geo_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
contra_harmonic_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
harmonic_filter(img,guassiannoise,poissonnoise,saltnoise,peppernoise,specklenoise,snpnoise)
