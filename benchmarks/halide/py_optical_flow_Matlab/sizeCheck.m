function [ A, B_in ] = sizeCheck( A, B_in, PYRE_NO )

%If image sizes are not integer multiples of maximum downsampling ratio,
%then this function resizes them

if sum( mod( size(A), 2^(PYRE_NO-1) ) ) ~= 0 
    A = imresize( A, size(A) - mod( size(A), 2^(PYRE_NO-1) ) );
end
   
if sum( mod( size(B_in), 2^(PYRE_NO-1) ) ) ~= 0 
    B_in = imresize( B_in, size(B_in) - mod( size(B_in), 2^(PYRE_NO-1) ) );
end

end
