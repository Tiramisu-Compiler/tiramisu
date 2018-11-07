function [ B ] = imWarp( flowHor, flowVer, Bin )

%This function warps B towards A

[x y] = meshgrid(1:size(Bin,2),1:size(Bin,1));

B = interp2(Bin, x+flowHor, y+flowVer, 'cubic');
B(isnan(B)) = Bin(isnan(B));

end
