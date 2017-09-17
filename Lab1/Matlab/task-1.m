

kidsImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/kids.tif'); 
cellImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/cell.tif'); 
mandiImage = imread('/Applications/MATLAB_R2015b.app/toolbox/images/imdata/mandi.tif'); 

a1=subplot(2,2,1); 
imagesc(kidsImage);
colormap(a1,jet); 

a2=subplot(2,2,2); 
imagesc(cellImage); 
colormap(a2,winter);

 
a3=subplot(2,2,3); 
imshow(mandiImage); 
colormap(a3,pink);



