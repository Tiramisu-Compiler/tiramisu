# RL_in_Search_space_exploration

This project contains the code for the Tiramisu autoscheduler using reinforcement learning for single computation. The important files are:
- The environment file: contains the representation of the Tiramisu search space as a OpenAI gym environement.
- The model file: contains the implementation of a feedforward network with two outputs: the policy output and the value output.
- The trainer file (ppoCustomModelEnhanced.py): contains the PPO trainer file provided by RLlib and configured according to our needs.
- The Tiramisu Program file (Tiramisu_program and Tiramisu_programLocal): Transforms the Tiramisu code into a class with attributes, there are two version, one to run on the cluster and one for a local run.