## Tutorials
--------------

This page has two tutorials: one for Tiramisu users and the other one for new Tiramisu compiler developers (useful for adding new features, optimizations and backends).

#### Tutorials for Tiramisu Users
- To be added soon.

#### Tutorials for Tiramisu Developpers

- [tutorial_01](tutorials/tutorial_01.cpp): A simple example of how to use Tiramisu (a simple assignment).
- [tutorial 02](tutorials/tutorial_02.cpp): blurxy.
- [tutorial 03](tutorials/tutorial_03.cpp): matrix multiplication.
- [tutorial 05](tutorials/tutorial_05.cpp): simple sequence of computations.
- [tutorial 06](tutorials/tutorial_06.cpp): reduction example.
- [tutorial 08](tutorials/tutorial_08.cpp): update example.
- [tutorial 09](tutorials/tutorial_09.cpp): complicated reduction/update example.

More examples can be found in the [tests](tests/) folder. Please check [test_descriptions.txt](tests/test_descriptions.txt) for a full list of examples for each Tiramisu feature.

#### Run Tutorials

To run all the tutorials, assuming you are in the build/ directory

    make tutorials
    
To run only one tutorial (tutorial_01 for example)

    make run_tutorial_01
    
This will compile and run the code generator and then the wrapper.

