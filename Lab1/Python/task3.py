import cv2

img=cv2.imread("test.jpg",1)

resized_img=cv2.resize(img, (200, 200))

cv2.imshow("Original",img)

cv2.imshow("Resized", resized_img)


cv2.imwrite("a.bmp", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
