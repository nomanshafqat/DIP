import cv2
import numpy as np
from matplotlib import pyplot as plt

'''Noman Shafqat '''

'''


NOTE: Before running one task please comment out the other tasks.


'''

# ------------------------------Task1------------------------------------------

image = cv2.imread('test.jpg', 0)
bright = cv2.add(image, 100)
dim = cv2.subtract(image, 100)
invert = cv2.subtract(255, image)

cv2.imshow("Original", image)
cv2.imshow("invert", invert)
cv2.imshow("Bright", bright)
cv2.imshow("DIM", dim)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------Task2a------------------------------------------ #


image = cv2.imread('test.jpg', 0)
cv2.imshow("original", image)

new = cv2.add(image, 1)
new = cv2.log(np.array(new).astype(float))

constants = [.1, 1, 2, 20, 30, 50, 60]
for c in constants:
    temp = cv2.multiply(new.astype('uint8'), c)
    cv2.imshow("c=" + str(c), temp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------Task2b------------------------------------------#


gammas = [.1, 0.3, .5, .7, 1, 1.1, 1.2, 1.5, 2]
# gammas=[1]


for gamma in gammas:
    temp = cv2.pow(image.astype(float), gamma)
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')

    print("gamma=", gamma, temp)
    cv2.imshow("Y=" + str(gamma), temp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------Task3------------------------------------------#


image = cv2.imread('test.bmp', 0)

y, x = image.shape[:2]
hist, bins = np.histogram(image.ravel(), 256, [0, 256])

hist = hist.cumsum()
newimg = (hist / (x * y))

newimg = cv2.multiply(newimg.astype(float), 255)
cdf = newimg.astype(float).flatten()

image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
image_equalized = image_equalized.reshape(image.shape)
equ = cv2.equalizeHist(image)

cv2.imshow("Yqualised", image_equalized.astype('uint8'))
cv2.imshow("Original", image.astype('uint8'))
cv2.imshow("Default function", equ.astype('uint8'))

plt.subplot(511)
plt.title('Original')
plt.hist(image.ravel(), 256, [0, 256])

plt.subplot(513)
plt.title('my algorithm')
plt.hist(image_equalized.ravel(), 256, [0, 256])

plt.subplot(515)
plt.title('Python library')
plt.hist(equ.ravel(), 256, [0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------Task4a------------------------------------------#

'''
The Equalised picture looks darker than the stretched picture
The histogram of the original image had 'less used' pixel ranges at the both ends

However, The stretched image changed colors and the darker pixels got more darker.
The the lighter pixels at 200 were stretched all the way over to 255. 

The normalised picture nicely spread the pixels all over the place and also it gave a longer range to the pixels where 
the density was more. 
'''

image = cv2.imread('cameraman.bmp', 0)
b = 255
a = 0

hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# print(bins)
# print(hist)
cumsum = hist.cumsum()
sum = hist.sum()

start = sum * .05
end = sum * .95
print(start, end)

c = np.where(cumsum.flatten() > start)[0][0]
d = np.where(cumsum.flatten() > end)[0][0]

print(a, b, c, d)

temp = cv2.subtract(image, int(c))
temp = cv2.multiply(temp, (255 / (d - c)))
temp = cv2.add(temp, a)

cv2.imshow("Streched", temp.astype("uint8"))
equ = cv2.equalizeHist(image)

cv2.imshow("Original", image.astype('uint8'))
cv2.imshow("Equalised", equ.astype('uint8'))

plt.subplot(5, 2, 1)
plt.title('Original')
plt.hist(image.ravel(), 256, [0, 256])

plt.subplot(522)
plt.title('Original')
plt.imshow(image.astype('uint8'), cmap='gray')

plt.subplot(525)
plt.title('streched')
plt.hist(temp.ravel(), 256, [0, 256])

plt.subplot(526)
plt.title('streched')
plt.imshow(temp.astype('uint8'), cmap='gray')

plt.subplot(5, 2, 9)
plt.title('equalised')
plt.hist(equ.ravel(), 256, [0, 256])

plt.subplot(5, 2, 10)
plt.title('equalised')
plt.imshow(equ.astype('uint8'), cmap='gray')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------Task4b------------------------------------------#


image = cv2.imread('cameraman.bmp', 0)
b = 255
a = 0

hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cumsum = hist.cumsum()
sum = hist.sum()

# ------------3%----------
start = sum * .03
end = sum * .97
c = np.where(cumsum.flatten() > start)[0][0]
d = np.where(cumsum.flatten() > end)[0][0]

print(a, b, c, d)
cv2.imshow("original", image)
temp = cv2.subtract(image, int(c))
temp = cv2.multiply(temp, (255 / (d - c)))
temp = cv2.add(temp, a)
cv2.imshow("Streched 3%", temp.astype("uint8"))

equ3 = cv2.equalizeHist(image)

# ----   12% ----



tart = sum * .12
end = sum * .88
c = np.where(cumsum.flatten() > start)[0][0]
d = np.where(cumsum.flatten() > end)[0][0]

print(a, b, c, d)
cv2.imshow("original", image)
temp = cv2.subtract(image, int(c))
temp = cv2.multiply(temp, (255 / (d - c)))
temp = cv2.add(temp, a)
cv2.imshow("Streched 12%", temp.astype("uint8"))

equ12 = cv2.equalizeHist(image)

# -plots-



plt.subplot(5, 1, 1)
plt.title('Original')
plt.hist(image.ravel(), 256, [0, 256])

plt.subplot(513)
plt.title('streched 3%')
plt.hist(equ3.ravel(), 256, [0, 256])

plt.subplot(5, 1, 5)
plt.title('streched 12%')
plt.hist(equ12.ravel(), 256, [0, 256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

'''Q:would it be desirable to only clip values from one end?
ANS: Yes, Since there are lots of pixels at the 7-15 range so increasing from 3% to 12% doesn't change a thing where as
the same change applied at the right side of the histogram has a drastic effect'''
