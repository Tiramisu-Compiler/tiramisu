- Two outer loops around a classical convolution.
    - The two outer loops represent time step and layers.
    - The code can be represented as
	for l in layer0 to layerN
	    for t in time0 to timeM
		classical-convolution;
