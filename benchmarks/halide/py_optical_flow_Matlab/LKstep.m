function  [us vs] = LKstep( It, Ix, Iy, H,  SizeBig )

%This function calculates one iteration of optical flow
us = zeros(size(It));
vs = zeros(size(It));

for i = 1+SizeBig : size(It,1)-SizeBig  
    for j = 1+SizeBig : size(It,2)-SizeBig       
      
        ix = Ix( i-SizeBig:i+SizeBig, j-SizeBig:j+SizeBig );        
        iy = Iy( i-SizeBig:i+SizeBig, j-SizeBig:j+SizeBig );                 
        it = It( i-SizeBig:i+SizeBig, j-SizeBig:j+SizeBig );
       
        b(1,1) = sum(sum( it .* ix ));                       
        b(2,1) = sum(sum( it .* iy ));        
        x = H(:,:,i,j) \ b; 
          
        us(i,j) = x(1);      
        vs(i,j) = x(2);
             
    end
end




end
