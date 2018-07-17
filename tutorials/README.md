# Tutorials

These tutorials are about the low level Tiramisu API.  Other tutorials that present the high level API are being prepared and will be published soon.  The low level API is verbose but allows full control over code generation and is supposed to be used mainly by the compiler developers, not the final users (but this is the only API that we provide for the moment).

- [Developers tutorial 01](developers/tutorial_01/tutorial_01.cpp): a simple assignment example (1D loop).
- [Developers tutorial 02](developers/tutorial_02/tutorial_02.cpp): another simple example (2D loop).
- [Developers tutorial 03](developers/tutorial_03/tutorial_03.cpp): a simple sequence of computations.
- [Developers tutorial 04A](developers/tutorial_04A/tutorial_04A.cpp): a matrix multiplication example.
- [Developers tutorial 04B](developers/tutorial_04B/tutorial_04B.cpp): an example of two successive matrix multiplications.
- [Developers tutorial 05](developers/tutorial_05/tutorial_05.cpp): an example of a reduction operation.
- [Developers tutorial 06](developers/tutorial_06/tutorial_06.cpp): an example of an update operation.
- [Developers tutorial 07](developers/tutorial_07/tutorial_07.cpp): a complicated example of reduction and an update.

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.

## Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial from the user tutorials (users/tutorial_01 for example)

    make run_users_tutorial_01

and

    make run_developers_tutorial_01
    
This will compile and run the code generator and then the wrapper.

