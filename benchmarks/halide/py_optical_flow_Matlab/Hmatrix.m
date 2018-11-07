function [ H] = Hmatrix( Ix, Iy, SizeBig, alfa )

%At each pyramid level, this function generates the Hessian matrix for the
%source image
 
H = zeros([2 2 size(Ix)-SizeBig]);

for i = 1+SizeBig : size(Ix,1)-SizeBig  
    for j = 1+SizeBig : size(Ix,2)-SizeBig     
    
        ix = Ix( i-SizeBig:i+SizeBig, j-SizeBig:j+SizeBig );        
        iy = Iy( i-SizeBig:i+SizeBig, j-SizeBig:j+SizeBig );
        H(1,1,i,j) = alfa+sum(sum( ix.^2 ));       
        H(2,2,i,j) = alfa+sum(sum( iy.^2 ));      
        H(1,2,i,j) = sum(sum( ix .* iy ));        
        H(2,1,i,j) = H(1,2,i,j);
                      
    end
end


end
