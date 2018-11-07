function [ Apyre, Bpyre, G, SizeBig ] = pyramidInitialize( A, B_in, PYRE_NO, winSize )

%This function forms image pyramids and initializes other variables

SizeBig = (winSize-1)/2;

Apyre{1} = A;
Bpyre{1} = B_in;

if PYRE_NO > 1
    for k = 2:PYRE_NO
        Apyre{k} = impyramid( Apyre{k-1}, 'reduce' );
        Bpyre{k} = impyramid( Bpyre{k-1}, 'reduce' );
    end
end

G = fspecial('gaussian',[3 3],1);

end
