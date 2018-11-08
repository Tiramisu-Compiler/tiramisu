% Code for dense pyramidal lucas-kanade extracted from
% https://www.mathworks.com/matlabcentral/fileexchange/22950-lucas-kanade-pyramidal-refined-optical-flow-implementation

clear all;
echo off;

SYNTHETIC_DATA = 1

if (SYNTHETIC_DATA)
       im1 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81;
              1, 2, 5, 10, 17, 26, 37, 50, 65, 82;
              4, 5, 8, 13, 20, 29, 40, 53, 68, 85;
              9, 10, 13, 18, 25, 34, 45, 58, 73, 90;
              16, 17, 20, 25, 32, 41, 52, 65, 80, 97;
              25, 26, 29, 34, 41, 50, 61, 74, 89, 106;
              36, 37, 40, 45, 52, 61, 72, 85, 100, 117;
              49, 50, 53, 58, 65, 74, 85, 98, 113, 130;
              64, 65, 68, 73, 80, 89, 100, 113, 128, 145;
              81, 82, 85, 90, 97, 106, 117, 130, 145, 162];

       im2 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81;
              1, 2, 5, 10, 17, 26, 37, 50, 65, 82;
              4, 5, 8, 13, 20, 29, 40, 53, 68, 85;
              9, 10, 13, 18, 25, 34, 45, 58, 73, 90;
              16, 17, 20, 25, 32, 41, 52, 65, 80, 97;
              25, 26, 29, 34, 41, 50, 61, 74, 89, 106;
              36, 37, 40, 45, 52, 61, 72, 85, 100, 117;
              49, 50, 53, 58, 65, 74, 85, 98, 113, 130;
              64, 65, 68, 73, 80, 89, 100, 113, 128, 145;
              81, 82, 85, 90, 97, 106, 117, 130, 145, 162];
    numLevels=2;    % levels number
    window=2;       % window size
    iterations=2;   % iterations number
else
    im1=single(rgb2gray(imread('/Users/b/Documents/src/MIT/tiramisu/utils/images/rgb.png')));
    im2=single(rgb2gray(imread('/Users/b/Documents/src/MIT/tiramisu/utils/images/rgb.png')));
    numLevels = 3;   % levels number
    window = 32;     % window size
    iterations = 3;  % iterations number
end

alpha = 0.001;  % regularization
hw = floor(window/2);

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pyramids creation
pyramid1 = im1;
pyramid2 = im2;

%init
for i=2:numLevels
    im1 = impyramid(im1, 'reduce');
    im2 = impyramid(im2, 'reduce');
    pyramid1(1:size(im1,1), 1:size(im1,2), i) = im1;
    pyramid2(1:size(im2,1), 1:size(im2,2), i) = im2;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Processing all levels
for p = 1:numLevels
    %current pyramid
    im1 = pyramid1(1:(size(pyramid1,1)/(2^(numLevels - p))), 1:(size(pyramid1,2)/(2^(numLevels - p))), (numLevels - p)+1);
    im2 = pyramid2(1:(size(pyramid2,1)/(2^(numLevels - p))), 1:(size(pyramid2,2)/(2^(numLevels - p))), (numLevels - p)+1);

    %init
    if p==1
        u=zeros(size(im1));
        v=zeros(size(im1));
    else  
        %resizing
        u = 2 * imresize(u,size(u)*2,'bilinear');   
        v = 2 * imresize(v,size(v)*2,'bilinear');
    end

    %refinment loop
    for r = 1:iterations
   
        u=round(u);
        v=round(v);
    
        %every pixel loop
        for i = 1+hw:size(im1,1)-hw
            for j = 1+hw:size(im2,2)-hw
                  patch1 = im1(i-hw:i+hw, j-hw:j+hw);

                  %moved patch
                  lr = i-hw+v(i,j);
                  hr = i+hw+v(i,j);
                  lc = j-hw+u(i,j);
                  hc = j+hw+u(i,j);

                  if (lr < 1)||(hr > size(im1,1))||(lc < 1)||(hc > size(im1,2))  
                  %Regularized least square processing
                  else
                  patch2 = im2(lr:hr, lc:hc);

                  fx = conv2(patch1, 0.25* [-1 1; -1 1]) + conv2(patch2, 0.25*[-1 1; -1 1]);
                  fy = conv2(patch1, 0.25* [-1 -1; 1 1]) + conv2(patch2, 0.25*[-1 -1; 1 1]);
                  ft = conv2(patch1, 0.25*ones(2)) + conv2(patch2, -0.25*ones(2));

                  Fx = fx(2:window-1,2:window-1)';
                  Fy = fy(2:window-1,2:window-1)';
                  Ft = ft(2:window-1,2:window-1)';
                  A = [Fx(:) Fy(:)];      

                  U = pinv(A) * -Ft(:);
                  u(i,j) = u(i,j) + U(1);
                  v(i,j) = v(i,j) + U(2);
                  end
            end
        end
    end
end

toc

u
v
