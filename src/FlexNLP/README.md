# Simple LSTM Accelerator Behavioral Interface
This code simulates the command interface of a LSTM accelerator including data-movement and computation commands. 
## Run main.cpp
```
g++ main.cpp
./a.out
```
## Hardware Architecture
* Hardware includes 
  * Input SRAM for x(t), h(t-1)
  * Weight SRAM for Wx, Wh, bx, bh
  * Output SRAM for h(t)
  * Cellstate SRAM for c(t)
* During computation, the hardware will update h(t) and c(t) according to input and weight

## C++ Files 
* LSTM.h: software lstm written in cpp
* LSTMAcc.h: lstm hardware that simulates command interface of the accelerator 
* main.cpp: testing code with verification
* Utils: Matrix and vector functions

## Notes
* The function of LSTM refers to [torch.nn.lstm] in Pytorch with the same order of gates (i, f, g, o). 
* Hidden state and cell state are treated different that hidden state input must be set explicitly for every timestep, but cell state is updated internally.
* As the hardware is performing computation in vector size of 16, the sizes of LSTM must be multiples of 16.   

[torch.nn.lstm]: https://pytorch.org/docs/master/generated/torch.nn.LSTM.html#torch.nn.LSTM
