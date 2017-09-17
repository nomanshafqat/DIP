
I = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/onion.png'); 

J = imresize(I, 0.5);

imwrite(J,'out.bmp')
