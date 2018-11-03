clear all;
image1 =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

image2 =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
w=2

im1 = im2double(image1);
im2 = im2double(image2);
C = [4, 6; 4, 6]
kernel1 = [-1 1;
           -1 1]
kernel1_90 = rot90(kernel1, 2)
kernel2 = [-1 -1;
            1 1]
kernel2_90 = rot90(kernel2, 2)

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
      p = pinv(A)
      nu = p*b;

      u(k)=nu(1);
      v(k)=nu(2);
end;

u
v


