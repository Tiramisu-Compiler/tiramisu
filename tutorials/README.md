# Tutorials

- [Developers tutorial 01](developers/tutorial_01/tutorial_01.cpp): a simple assignment example (1D loop).
- [Developers tutorial 02](developers/tutorial_02/tutorial_02.cpp): another simple example (2D loop).
- [Developers tutorial 03](developers/tutorial_03/tutorial_03.cpp): a simple sequence of computations.
- [Developers tutorial 04A](developers/tutorial_04A/tutorial_04A.cpp): a matrix multiplication example.
- [Developers tutorial 04B](developers/tutorial_04B/tutorial_04B.cpp): an example of two successive matrix multiplications.
- [Developers tutorial 04gpu](developers/tutorial_04gpu/tutorial_04gpu.cpp): an example of a matrix multiplication mapped to GPU.

The following tutorials are written using the low level Tiramisu API (most users do not need to learn the low level API).

- [Developers tutorial 05](developers/tutorial_05/tutorial_05.cpp): an example of a reduction operation.
- [Developers tutorial 06](developers/tutorial_06/tutorial_06.cpp): an example of an update operation.
- [Developers tutorial 07](developers/tutorial_07/tutorial_07.cpp): a complicated example of reduction and an update.
- [Developers tutorial 08](developers/tutorial_08/tutorial_08.cpp): tutorial 02 written using the low level Tiramisu API (this API allows full control over code generation).

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.

## Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial from the user tutorials (users/tutorial_01 for example)

    make run_users_tutorial_01

and

    make run_developers_tutorial_01
    
This will compile and run the code generator and then the wrapper.

