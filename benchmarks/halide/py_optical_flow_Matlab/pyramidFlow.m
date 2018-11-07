function [ u v H ] = pyramidFlow( A, B_in, winSize, ITER_NO, PYRE_NO )

%This function find the optical flow from A to B using pyramid
%representation and iteration
%To handle the pixels on the borders, a smaller window is used

%PYRE_NO : total number of pyramid levels
%ITER_NO : number of iterations at each pyramid level
%G : Gaussian kernel for smoothing

[ A, B_in ] = sizeCheck( A, B_in, PYRE_NO );
    
[ Apyre, Bpyre, HalfWindow ] = pyramidInit( A, B_in, PYRE_NO, winSize );


for p = PYRE_NO:-1:1
    fprintf('Pyramid level: %d\n',p)
    
    A = imReflect( Apyre{p}, HalfWindow );
    B = Bpyre{p};
    
   
    if p == PYRE_NO
        u = zeros(size( Apyre{p} ));
        v = zeros(size( Apyre{p} ));
        flag_ = 0;
    end
    
    %Generating the Hessian matrices for this level
    
      
    for k = 1:ITER_NO        
        fprintf('Iteration no: %d\n',k)
        
        %Calculate a single LK step
        
        if flag_ ~= 0
            B = imWarp( u, v, Bpyre{p} );
        else
            flag_ = 1;
        end
   
        B_ref = imReflect(B, HalfWindow);
        
        [Ix Iy] = gradient( B_ref );
        H  = Hmatrix( Ix, Iy, HalfWindow, 0.001 );
        
        It = A - B_ref;
        
        [us vs] = LKstep(It, Ix, Iy, H, HalfWindow);
                 
        us = us(HalfWindow+1:size(us,1)-HalfWindow, HalfWindow+1:size(us,2)-HalfWindow);
        vs = vs(HalfWindow+1:size(vs,1)-HalfWindow, HalfWindow+1:size(vs,2)-HalfWindow);   
       
        u = u + us;
        v = v + vs;
    end
       
    if p ~= 1 
        u = 2 * imresize(u,size(u)*2,'bilinear');
        v = 2 * imresize(v,size(v)*2,'bilinear');
    end

    
end


end
