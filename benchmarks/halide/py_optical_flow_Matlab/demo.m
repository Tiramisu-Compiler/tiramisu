%convert images to double and grayscale
I1 = impyramid( imInit('car1.png'), 'reduce' );
I2 = impyramid( imInit('car2.png'), 'reduce' );

%%

[flowHor flowVer] = pyramidFlow(I1, I2, 5, 3, 3);  %pyramidFlow( I1, I2, winSize, ITER_NO, PYRE_NO )

%show the output flows 
imS(flowHor,1,[-10 10]) 
imS(flowVer,2,[-10 10])

%%
%show the warped image to check the quality of registration
Iw = imWarp(flowHor, flowVer, I2);
imS(I1,10)
imS(Iw,11)
imS(I2,12)
