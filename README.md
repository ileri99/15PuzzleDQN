# 15PuzzleDQN
Using a neural network to solve the 15 puzzle

This is a continuation of the 8 puzzle problem, except instead of 100,000 states we're now dealing with 10 trillion states. As a result the agent requires days of training before it starts occasionally solving the game. I plan on speeding up the training process by optimising my code to run on a GPU. 

Because of the large state space I will need to split the program into separate files to store the game, and the DQN agent. This will make it easier to design a script to update the weights of the network. I will also be able to create a render of the game, to make it easier to watch the agent solve a random puzzle.

This code will be updated with an agent that has already been trained to reliably solve the game in under 100 moves (under 80 is the optimal zone). 
