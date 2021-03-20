# Reactive-Navigation-Under-a-Fuzzy-Rules-Based-Scheme-and-Reinforcement-Learning
# Reactive-Navigation-Under-a-Fuzzy-Rules-Based-Scheme-and-Reinforcement-Learning


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Structure](#structure)


## General info
This is a navigation approach proposal for mobile robots. This project includes two modules, the path planning and the decision-making modules. 
The goal is that a robot moves to a destination in a static enviroment avoiding obstacles, and it decides when to go the destination or the battery chaging station.
The artificial potential fields method is used for path planning, while fuzzy q-learning is used for decision-making.

	
## Technologies
Project is created with:
* Python 3.8
	
## Setup
To run this project, install it locally python 3.8. the project was created using PyCharm Community Edition, so you can install it or if rather you can execute the simulations from a terminal on linux.

## Structure

The project has five folders where you will find the sources files to execute the examples, and the simulations. The proposal was compared with othe approaches of 
reinforcement learning as q-learning and SARSA, also some deterministic methods were used as threshold and the fuzzy inference systems.

* Examples
  * Contains some examples about the implemetation of the classes as the fuzzy classes, and path planning.
* Fuzzy
  * Contains the classes to set the fuzzy sets, and the inference system.
* Learning
  * Contains the classes used to make an implementation with FQL, QL, and SARSA.
* Planning
  * Contains the classes used for path planning.
* Simulations
  * Contains the files for the simulations executed for the paper titled "Reactive Navigation Under a Fuzzy Rules Based Scheme and Reinforcement Learning", which is on revision
