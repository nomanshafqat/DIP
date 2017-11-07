import numpy as np
import cv2
import matplotlib.pyplot as plt

"Noman Shafqat SE5A 111572"

def butterworth(ft,cuttof,order,type="low"):

    filter=np.zeros(ft.shape)
    midx=filter.shape[0]/2
    midy=filter.shape[1]/2

    if(type=="low"):

        for i in range(0,filter.shape[0]):
            for j in range(0,filter.shape[1]):
                distance=np.sqrt((i - midx) ** 2 + (j - midy) ** 2)
                filter[i][j]=1/(1+np.power(distance/cuttof,order))
            #print(filter[i])

    else:
        for i in range(0,filter.shape[0]):
            for j in range(0,filter.shape[1]):
                distance=np.sqrt((i - midx) ** 2 + (j - midy) ** 2)
                filter[i][j]=(1/(1+np.power(distance/cuttof,order)))
                filter[i][j]=1- filter[i][j]
            #print(filter[i])

    return filter


def ideal(ft,cuttof,type="low"):

    filter=np.zeros(ft.shape)
    midx=filter.shape[0]/2
    midy=filter.shape[1]/2

    if(type=="low"):

        for i in range(0,filter.shape[0]):
            for j in range(0,filter.shape[1]):
                distance=np.sqrt((i - midx) ** 2 + (j - midy) ** 2)
                if(distance<cuttof):
                    filter[i][j]=1
                else:
                    filter[i][j]=0

            #print(filter[i])

    else:
        for i in range(0,filter.shape[0]):
            for j in range(0,filter.shape[1]):
                distance=np.sqrt((i - midx) ** 2 + (j - midy) ** 2)
                if (distance < cuttof):
                    filter[i][j] = 0
                else:
                    filter[i][j] = 1

            #print(filter[i])

    return filter


def guassian(ft,cuttoff,type="low"):
    filter = np.zeros(ft.shape)
    midx = filter.shape[0] / 2
    midy = filter.shape[1] / 2

    if (type == "low"):

        for i in range(0, filter.shape[0]):
            for j in range(0, filter.shape[1]):
                distance = ((i - midx) ** 2 + (j - midy) ** 2)
                filter[i][j] = np.exp(-distance/(2*cuttoff))
                # print(filter[i])

    else:
        for i in range(0, filter.shape[0]):
            for j in range(0, filter.shape[1]):
                distance = np.sqrt((i - midx) ** 2 + (j - midy) ** 2)
                distance = ((i - midx) ** 2 + (j - midy) ** 2)
                filter[i][j] = np.exp(-distance / (2 * cuttoff))
                filter[i][j] = 1 - filter[i][j]
                # print(filter[i])

    return filter


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype


def doubetounit(im):
    return np.multiply(im,255).astype('unit')


def task1():
    image = cv2.imread("Cap.jpg", 0)
    fft_orig = np.fft.fftshift(np.fft.fft2(image))

    recon_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_orig)))

    low=butterworth(fft_orig,30,6,"low")
    high=butterworth(fft_orig,30,6,"high")

    lowpass_filtered=np.multiply(low,fft_orig)
    lowpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(lowpass_filtered)))

    highpass_filtered=np.multiply(high,fft_orig)
    highpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(highpass_filtered)))

    print(recon_image)

    plt.subplot(321)
    plt.imshow( abs(np.log10(fft_orig)) , cmap='gray')
    plt.axis('off')


    plt.subplot(322)
    plt.imshow( abs(recon_image), cmap='gray')
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(high, cmap='gray')
    plt.axis('off')

    plt.subplot(324)
    plt.title("ButterWorth highpass")
    plt.imshow( highpass_filtered.astype("uint"), cmap='gray')
    plt.axis('off')

    plt.subplot(325)

    plt.imshow(low, cmap='gray')
    plt.axis('off')

    plt.subplot(326)
    plt.title("ButterWorth lowpass")

    plt.imshow( lowpass_filtered.astype("uint"), cmap='gray')
    plt.axis('off')

    plt.show()


def task2():
    image = cv2.imread("Cameraman.bmp", 0)
    image=im2double(image)
    fft_orig = np.fft.fftshift(np.fft.fft2(image))
    recon_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_orig)))

    max=np.max(abs(fft_orig))
    min=np.min(abs(fft_orig))

    print(max,min)

    low = ideal(fft_orig, 30 ,"low")
    high = ideal(fft_orig, 30,"high")

    lowpass_filtered = np.multiply(low, fft_orig)
    lowpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(lowpass_filtered)))

    highpass_filtered = np.multiply(high, fft_orig)
    highpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(highpass_filtered)))
    #print(highpass_filtered)

    plt.subplot(421)
    plt.imshow(abs(np.log10(fft_orig)), cmap='gray')
    plt.axis('off')
    #print(recon_image)
    plt.subplot(422)
    plt.imshow(recon_image, cmap='gray')
    plt.axis('off')


    plt.subplot(423)
    plt.imshow(high, cmap='gray')
    plt.axis('off')

    plt.subplot(424)
    plt.title("Ideal highpass")

    plt.imshow(highpass_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(425)
    plt.imshow(low, cmap='gray')
    plt.axis('off')

    plt.subplot(426)
    plt.title("Ideal lowpass")

    plt.imshow(lowpass_filtered, cmap='gray')
    plt.axis('off')

    plt.show()


def task3():
    image = cv2.imread("Cameraman.bmp", 0)
    image = im2double(image)
    fft_orig = np.fft.fftshift(np.fft.fft2(image))
    recon_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_orig)))

    max = np.max(abs(fft_orig))
    min = np.min(abs(fft_orig))

    print(max, min)

    low = guassian(fft_orig, 30, "low")
    high = guassian(fft_orig, 30, "high")

    lowpass_filtered = np.multiply(low, fft_orig)
    lowpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(lowpass_filtered)))

    highpass_filtered = np.multiply(high, fft_orig)
    highpass_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(highpass_filtered)))
    # print(highpass_filtered)
    plt.subplot(421)
    plt.imshow(abs(np.log10(fft_orig)), cmap='gray')
    plt.axis('off')
    # print(recon_image)
    plt.subplot(322)
    plt.imshow(recon_image, cmap='gray')
    plt.axis('off')

    plt.subplot(323)
    plt.imshow(high, cmap='gray')
    plt.axis('off')

    plt.subplot(324)
    plt.title("Guassian highpass")

    plt.imshow(highpass_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(325)
    plt.imshow(low, cmap='gray')
    plt.axis('off')

    plt.subplot(326)
    plt.title("Guassian lowpass")

    plt.imshow(lowpass_filtered, cmap='gray')
    plt.axis('off')

    plt.show()

task1()
task2()
task3()
