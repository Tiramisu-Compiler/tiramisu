function [ Io ] = imInit( str )

Io = im2double( rgb2gray( imread(str) ) );

end
