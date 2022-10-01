# Entropy Maximized Multi-Robot Patrolling<br/>with Steady State Distribution Approximation
## Introduction
* This repository serves as a complement to our work on multi-robot patrolling problem.
* This is a joint program among A*STAR, MIT and NUS. (© 2022 A\*STAR. All rights reserved.)
* The paper was submitted to IEEE ICRA 2023. 

## Dependencies
* Python 3.8
* NetworkX 2.8.4
* Numpy
* Matplotlib
* PyTorch (GPU Acceleration is recommended)
* tqdm

## Description
* We integrate our simulation toolbox into a Python package called `emp`. 
* `opt.py` realizes the centralized optimizer for given environment and MRS.
* `env.py`, `robot.py` and `policy.py` implement entities that are involved in optimization and execution process.
* `main.py` claims legal arguments of command line input and the procedure of our algorithm.

## Usage
If you want to optimize a MRS of 4 robots in the MUSEUM environment with learning rate equals to 1e-3 for 5000 epochs and test it for 1000000 steps, as well as real-time feedbacks and final outputs:
```
python main.py --env MUSEUM --size_MRS 4 --epoch 5000 --step 1000000 --lr 1e-3 --verbose --output
```
For more operations, please refer to `parse_opt()` in `main.py`.
