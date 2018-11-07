function [] = imS( Iin, figNo, varargin )

if nargin == 2
    figure(figNo), imshow(Iin)
else
    figure(figNo), imshow(Iin, varargin{:})
end

end
