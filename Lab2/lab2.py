from PIL import Image
import numpy as np
import cv2



image = cv2.imread('test.jpg',0)
print(image)
cv2.imshow('image', image)

orig_arr = np.array(image)
i, j = orig_arr.shape

image_file = np.invert(orig_arr)

image_file=image_file.astype(int)
print(image_file)


for x in range(0, i):
    for y in range(0, j):
        if(image_file[x][y]<200):
            image_file[x][y]=0
        #print(image_file[x][y], end="\t"),
    #print(end="\n"),
#exit()

nextlabel = 0
back = 0
dict = {}
max = 0
for x in range(0, i):
    for y in range(0, j):

        if image_file[x][y] != back:

            if x != 0 and y != 0:

                left = image_file[x - 1][y]
                top = image_file[x][y - 1]
                #print("X=", x, "Y=", y, "====", image_file[x][y], "  left====", left, "top===", top)

                if top != back and left != back:
                    if top < left:
                        image_file[x][y] = top
                        if(left  in dict.keys()):
                            if(dict[left]!=top):
                                print("gghappla in here")

                            dict[left]=top
                        #print("Assiggned=top=", top)
                    elif left < top:

                        image_file[x][y] = left

                        if (top in dict.keys()):
                            if(dict[top]!=left):
                                temp = dict[top]
                                while temp in dict.keys():
                                    print(temp)
                                    temp = dict[temp]

                                top=temp
                                print("gghappla in here")

                        if(top!=left):
                            dict[top]=left
                        #print("Assiggned=left=", left)
                    elif left == top:
                        image_file[x][y] = top
                        #print("Assiggned=any=", left)

                elif top != back:
                    image_file[x][y] = top
                    #print("Assiggned=top=", top)

                elif left != back:
                    image_file[x][y] = left
                    #print("Assiggned=left=", left)


                else:
                    nextlabel += 1
                    image_file[x][y] = nextlabel

            elif x == 0 and y != 0:
                #print("X=", x, "Y=", y)

                top = image_file[x][y - 1]
                if top != back:
                    image_file[x][y] = top
                else:
                    nextlabel += 1
                    image_file[x][y] = nextlabel

            elif x != 0 and y == 0:
                #print("X=", x, "Y=", y)

                left = image_file[x - 1][y]
                if left != back:
                    image_file[x][y] = left
                else:
                    nextlabel += 1
                    image_file[x][y] = nextlabel



            else:
                #print("X=", x, "Y=", y)

                nextlabel += 1
                image_file[x][y] = nextlabel

print("\n",nextlabel,"\n")
for x in range(0, i):
    for y in range(0, j):
        if(image_file[x][y] in dict.keys()):
            temp=image_file[x][y]
            while temp  in dict.keys():
                temp=dict[temp]

            image_file[x][y]=(temp+100)/3
#for x in range(0, i):
    #for y in range(0, j):
        #print(image_file[x][y], end="\t\t"),
    #print(end="\n"),
print(dict.items())

cv2.imwrite('messigray.png', image_file)
