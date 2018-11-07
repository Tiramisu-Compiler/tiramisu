function [ Io ] = imReflect( I, bordSize )

%bordSize : number of reflected pixels from each side of the image

Io = zeros(size(I)+2*bordSize);
Io(bordSize+1:size(I,1)+bordSize, bordSize+1:size(I,2)+bordSize) = I;

for j = 1:bordSize 
    Io(bordSize+1 : size(I,1)+bordSize, j) = I(1:size(I,1), bordSize+1-j);
end

for j = size(I,2)+1+bordSize:size(I,2)+2*bordSize 
    Io(bordSize+1 : size(I,1)+bordSize, j) = I(1:size(I,1), 2*size(I,2)+bordSize+1-j);
end

for i = 1:bordSize 
    Io(i, 1:size(Io,2)) = Io(2*bordSize+1-i, 1:size(Io,2));
end

for i = size(I,1)+1+bordSize:size(I,1)+2*bordSize 
    Io(i, 1:size(Io,2)) = Io(2*size(I,1)+1+2*bordSize-i, 1:size(Io,2));
end

end
