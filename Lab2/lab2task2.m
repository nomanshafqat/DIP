img=imread('lab2-task2-image.jpg');

[x1,y1,z1] = size(img);
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);
x1
y1
z1
for i=1:x1
    for j=1:y1
        b = img(i,j,1);
        g = img(i,j,2);
        r = img(i,j,3);
         if  (r >= 20) && (r<= 55) && (g >= 15) && (g <= 45) (b >= 200) && (b <= 255);
                r=0 ;g=255 ;b=255;
                img(i,j,1)=b;
                img(i,j,2)=g;
                img(i,j,3)=r;
         elseif  (r >= 70) && (r<= 80) && (g >= 150) && (g <= 255) (b >= 30) && (b <= 40);
                r=255 ;g=255 ;b=0;
                img(i,j,1)=b;
                img(i,j,2)=g;
                img(i,j,3)=r;
         elseif  (r >= 200) && (r<= 255) && (g >= 65) && (g <= 75) (b >= 55) && (b <= 65);
                r=255 ;g=0 ;b=255;
                img(i,j,1)=b;
                img(i,j,2)=g;
                img(i,j,3)=r;
         end
            
    end
end

imwrite(img,'sd.png');
