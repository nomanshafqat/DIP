img = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/onion.png'); 

red = img(:,:,1);
green = img(:,:,2); 
blue = img(:,:,3);

zero = zeros(size(img, 1), size(img, 2));
red = cat(3, red, zero, zero);
green = cat(3, zero, green, zero);
blue = cat(3, zero, zero, blue);

subplot(2,2,1);
imshow(red)

subplot(2,2,2);
imshow(green)

subplot(2,2,3);
imshow(blue)
