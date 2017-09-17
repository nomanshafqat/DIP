from PIL import Image

im = Image.open("test.jpg")

print("---Numbers and other data types----")

print(type(-75))
print(type(5.0))
print(type(12345678901))

print("---Strings----")

print(" This is a string ")
print("This is a string, too")
print(type("This is a string "))

print("---Lists and tuples----")

print([1, 3, 4, 1, 6])
print(type([1, 3, 4, 1, 6]))
print(type((1, 3, 2)))

print("--- The range function----")

print(range(17))
print(range(1, 10))
print(range(-6, 0))
print(range(1, 10, 2))
print(range(10, 0, -2))

print("---Operators----")

x = 2 + 2
print(x)
x = 380.5

print(x)
y = 2 * x
print(y)

print("---DECISIONS----")

x = 1
if x > 0:
    print(" Friday is wonderful")
else:
    print(" Monday so perform LAB")
print(" Have a good weekend")

print("---LOOPS--FOR--")

for i in [2, 4, 6, 0]:
    print(i)

print("---LOOPS--while--")
n = 0
while n < 10:
    print(n)
    n = n + 3



'''---------------TASK2-------------'''

from PIL import Image
import numpy as np
import cv2
image = Image.open("test.jpg")
orig_arr = np.array(image)


red = orig_arr.copy()
blue = orig_arr.copy()
green = orig_arr.copy()

red[:, :, 1] = 0
red[:, :, 2] = 0

Image._show(Image.fromarray(red))

blue[:, :, 0] = 0
blue[:, :, 2] = 0
Image._show(Image.fromarray(blue))


green[:, :, 0] = 0
green[:, :, 1] = 0
Image._show(Image.fromarray(green))



'''---------------TASK3-------------'''



import cv2

img=cv2.imread("test.jpg",1)

resized_img=cv2.resize(img, (200, 200))

cv2.imshow("Original",img)

cv2.imshow("Resized", resized_img)


cv2.imwrite("a.bmp", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
