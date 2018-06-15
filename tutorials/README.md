## Tutorials
--------------

This page has two tutorials: one for Tiramisu users and the other one for new Tiramisu compiler developers (useful for adding new features, optimizations and backends).

#### Tutorials for Tiramisu Users
If you want to learn how to write Tiramisu code, this is the right tutorial
for you.

- [users_tutorial_01](tutorials/users/tutorial_01/tutorial_01.cpp): Tiramisu hello world (assign 0 to an array).


#### Tutorials for Tiramisu Developpers

If you want to contribute to the Tiramisu compiler itself (add new
optimizations, new backend, improve the front-end, ...), the following
tutorials will help you learn more about the internal representation
of Tiramisu.

- [developers_tutorial_01](tutorials/developers/tutorial_01/tutorial_01.cpp): A simple example of how to use the low level Tiramisu API (a simple assignment).
- [developers_tutorial 02](tutorials/developers/tutorial_02/tutorial_02.cpp): blurxy using the low level Tiramisu API.
- [developers_tutorial 03](tutorials/developers/tutorial_03/tutorial_03.cpp): matrix multiplication using the low level Tiramisu API.
- [developers_tutorial 05](tutorials/developers/tutorial_05/tutorial_05.cpp): simple sequence of computations using the low level Tiramisu API.
- [developers_tutorial 06](tutorials/developers/tutorial_06/tutorial_06.cpp): reduction example using the low level Tiramisu API.
- [developers_tutorial 08](tutorials/developers/tutorial_08/tutorial_08.cpp): update example using the low level Tiramisu API.
- [developers_tutorial 09](tutorials/developers/tutorial_09/tutorial_09.cpp): complicated reduction/update example using the low level Tiramisu API.

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.

#### Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial from the user tutorials (users/tutorial_01 for example)

    make run_users_tutorial_01

and

    make run_developers_tutorial_01
    
This will compile and run the code generator and then the wrapper.

