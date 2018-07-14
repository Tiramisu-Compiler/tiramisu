# Tutorials

This page has two types of tutorials: Tutorials written for all Tiramisu users and tutorials written only for compiler developers (those who will adding new features to the Tiramisu compiler, new optimizations, backends, ...).

## Tutorials for Tiramisu Users
If you want to learn how to write Tiramisu code, this is the right tutorial
for you.

- [Users tutorial 01](users/tutorial_01/tutorial_01.cpp): A simple example of how to use the high level Tiramisu API (a simple loop with an assignment).


## Tutorials for Tiramisu Developers (Low Level Tiramisu API)

If you want to contribute to the Tiramisu compiler (add new
optimizations, new backend, improve the front-end, ...), the following
tutorials will help you learn more about the internal representation
of Tiramisu.

- [Developers tutorial 01](developers/tutorial_01/tutorial_01.cpp): A simple example of how to use the low level Tiramisu API (a simple assignment).
- [Developers tutorial 02](developers/tutorial_02/tutorial_02.cpp): blurxy using the low level Tiramisu API.
- [Developers tutorial 03](developers/tutorial_03/tutorial_03.cpp): matrix multiplication using the low level Tiramisu API.
- [Developers tutorial 05](developers/tutorial_05/tutorial_05.cpp): simple sequence of computations using the low level Tiramisu API.
- [Developers tutorial 06](developers/tutorial_06/tutorial_06.cpp): reduction example using the low level Tiramisu API.
- [Developers tutorial 08](developers/tutorial_08/tutorial_08.cpp): update example using the low level Tiramisu API.
- [Developers tutorial 09](developers/tutorial_09/tutorial_09.cpp): complicated reduction/update example using the low level Tiramisu API.

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.

## Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial from the user tutorials (users/tutorial_01 for example)

    make run_users_tutorial_01

and

    make run_developers_tutorial_01
    
This will compile and run the code generator and then the wrapper.

