clear all;
echo off;

SYNTHETIC_DATA = 0

if (SYNTHETIC_DATA)
image1 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81;
          1, 2, 5, 10, 17, 26, 37, 50, 65, 82;
          4, 5, 8, 13, 20, 29, 40, 53, 68, 85;
          9, 10, 13, 18, 25, 34, 45, 58, 73, 90;
          16, 17, 20, 25, 32, 41, 52, 65, 80, 97;
          25, 26, 29, 34, 41, 50, 61, 74, 89, 106;
          36, 37, 40, 45, 52, 61, 72, 85, 100, 117;
          49, 50, 53, 58, 65, 74, 85, 98, 113, 130;
          64, 65, 68, 73, 80, 89, 100, 113, 128, 145;
          81, 82, 85, 90, 97, 106, 117, 130, 145, 162];

image2 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81;
          1, 2, 5, 10, 17, 26, 37, 50, 65, 82;
          4, 5, 8, 13, 20, 29, 40, 53, 68, 85;
          9, 10, 13, 18, 25, 34, 45, 58, 73, 90;
          16, 17, 20, 25, 32, 41, 52, 65, 80, 97;
          25, 26, 29, 34, 41, 50, 61, 74, 89, 106;
          36, 37, 40, 45, 52, 61, 72, 85, 100, 117;
          49, 50, 53, 58, 65, 74, 85, 98, 113, 130;
          64, 65, 68, 73, 80, 89, 100, 113, 128, 145;
          81, 82, 85, 90, 97, 106, 117, 130, 145, 162];
   w=2;
else
   image1_rgb = imread('/Users/b/Documents/src/MIT/tiramisu/utils/images/rgb.png');
   image1 = rgb2gray(image1_rgb);
   image2_rgb = imread('/Users/b/Documents/src/MIT/tiramisu/utils/images/rgb.png');
   image2 = rgb2gray(image2_rgb);
   w=128;
end


im1 = im2double(image1);
im2 = im2double(image2);
C = [500, 400;
     800, 900;
 	 200, 400;
     400, 200;
 	 400, 500;
     800, 200;
     200, 900;
     900, 200];
 
kernel1 = [-1 1;
           -1 1];
kernel1_90 = rot90(kernel1, 2);
kernel2 = [-1 -1;
            1 1];
kernel2_90 = rot90(kernel2, 2);

tic

Ix_m = conv2(im1, kernel1, 'valid');
Iy_m = conv2(im1, kernel2 , 'valid'); % partial on y
It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); % partial on t
u = zeros(length(C),1);
v = zeros(length(C),1);
for k = 1:length(C(:,2))
      i = C(k,2);
      j = C(k,1);
      Ix = Ix_m(i-w:i+w, j-w:j+w);
      Iy = Iy_m(i-w:i+w, j-w:j+w);
      It = It_m(i-w:i+w, j-w:j+w);

      Ix = Ix(:); % flatten the IX 2D array into a vector
      Iy = Iy(:);
      b = -It(:); % get b here

      A = [Ix Iy]; % get A here
      p = pinv(A);
      nu = p*b;

      u(k)=nu(1);
      v(k)=nu(2);
end;

toc

%u
%v

