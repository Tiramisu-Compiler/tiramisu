function [ Apyre, Bpyre, halfWindow ] = pyramidInit( A, B_in, PYRE_NO, winSize )

%This function forms image pyramids and initializes other variables

halfWindow = (winSize-1)/2;
G = fspecial('gaussian',[3 3],1);

Apyre{1} = conv2( A, G, 'same' );
Bpyre{1} = conv2( B_in, G, 'same' );

if PYRE_NO > 1
    for k = 2:PYRE_NO
        Apyre{k} = impyramid( Apyre{k-1}, 'reduce' );
        Bpyre{k} = impyramid( Bpyre{k-1}, 'reduce' );
    end
end

end
