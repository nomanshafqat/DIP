

kidsImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/kids.tif'); 
cellImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/cell.tif'); 
mandiImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/mandi.tif'); 

a1=subplot(3,3,1); 
imagesc(kidsImage);
colormap(a1,jet); 

a2=subplot(3,3,2); 
imagesc(cellImage); 
colormap(a2,winter);

 
a3=subplot(3,3,3); 
imshow(mandiImage); 
colormap(a3,pink);

%task2


img = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/onion.png'); 

red = img(:,:,1);
green = img(:,:,2); 
blue = img(:,:,3);

zero = zeros(size(img, 1), size(img, 2));
red = cat(3, red, zero, zero);
green = cat(3, zero, green, zero);
blue = cat(3, zero, zero, blue);

subplot(3,3,7);
imshow(red)

subplot(3,3,8);
imshow(green)

subplot(3,3,9);
imshow(blue)



I = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/onion.png'); 

J = imresize(I, 0.5);

imwrite(J,'out.bmp')

