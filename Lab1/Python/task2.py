'''moni hAS been trolled , filled the form late and has been forced to take distributed computing.Bechara aa k DIP ki lab mein beth k
happens moni , mujhse DIP le lo aur NS de do :)'''

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




