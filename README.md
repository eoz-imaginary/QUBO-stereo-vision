# A partial implementation of "A QUBO Formulation of the Stereo Matching Problem for D-Wave Quantum Annealers" by W. Cruz-Santos et al. 

This program implements the QUBO formulation from the above paper, and uses the D-Wave Simulated Quantum Annealer (D-Wave Neal) to compute the max-flow min-cut of a NetworkX DiGraph. 

To run: first install the required Python dependencies with `pip install -r requirements.txt`, and ensure that you are using Python 3.8. Python 3.9+ should theoretically work as well, but has not been tested. Then, in a terminal window run `python main.py` to generate a simple graph (5 nodes), create the Q-matrix for that graph, and use D-Wave's Simulated Annealer to find the max-flow min-cut of that graph. The final result will be saved to `qubo_graphcut_results.png`. A sample output for the 5-node graph has been provided. 
