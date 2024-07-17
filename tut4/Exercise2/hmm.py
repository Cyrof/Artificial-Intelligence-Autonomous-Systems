import numpy as np

def hmm_probability(observations, states, start_probability, transition_probability, emission_probability):
    """
    Calculate the probability of observations given HMM parameters.
    
    Args:
    - observations: List of observed states or observations.
    - states: List of possible states.
    - start_probability: Initial state probabilities (dictionary).
    - transition_probability: Transition probabilities (nested dictionary).
    - emission_probability: Emission probabilities (nested dictionary).
    
    Returns:
    - prob: Probability of observing the sequence of observations.
    """
    num_states = len(states)
    num_observations = len(observations)
    
    # Initialize the forward probabilities matrix
    forward_prob = np.zeros((num_states, num_observations))

    # Initialisation step
    for s in range(num_states):
        forward_prob[s, 0] = start_probability[states[s]] * emission_probability[states[s]][observations[0]]

    # Recursion step
    for t in range(1, num_observations):
        for x in range(num_states):
            forward_prob[s, t] = sum(forward_prob[s_prev, t-1] * transition_probability[states[s_prev]][states[s]] for s_prev in range(num_states)) * emission_probability[states[s]][observations[t]]
    
    # Termination step 
    prob = sum(forward_prob[s, num_observations-1] for s in range(num_states))
    
    return prob


def viterbi(observations, states, start_probability, transition_probability, emission_probability):
    """
    Find the optimal sequence of hidden states using the Viterbi algorithm.
    
    Args:
    - observations: List of observed states or observations.
    - states: List of possible states.
    - start_probability: Initial state probabilities (dictionary).
    - transition_probability: Transition probabilities (nested dictionary).
    - emission_probability: Emission probabilities (nested dictionary).
    
    Returns:
    - best_path: Optimal sequence of hidden states.
    """
    num_states = len(states)
    num_observations = len(observations)

    # Initialise the Viterbi and backpointer matrices
    viterbi_prob = np.zeros((num_states, num_observations))    
    backpointer = np.zeros((num_states, num_observations), dtype=int)

    # intialisation step 
    for s in range(num_states):
        viterbi_prob[s, 0] = start_probability[states[s]] * emission_probability[states[s]][observations[0]]
        backpointer[s, 0 ] = 0    

    # Recursion step 
    for t in range(1, num_observations):
        for x in range(num_states):
            transition_prob = [viterbi_prob[s_prev, t-1] * transition_probability[states[s_prev]][states[s]] for s_prev in range(num_states)]
            max_transition_prob = max(transition_prob)
            viterbi_prob[s, t] = max_transition_prob * emission_probability[states[s]][observations[t]]
            backpointer[s, t] = np.argmax(transition_prob)
    
    # Termination step
    best_path_prob = max(viterbi_prob[:, num_observations-1])
    best_last_state = np.argmax(viterbi_prob[:, num_observations-1])

    # Path backtracking
    best_path = [best_last_state]
    for t in range(num_observations-1, 0, -1):
        best_path.insert(0, backpointer[best_path[0], t])
        
    best_path_states = [states[state] for state in best_path]
    
    return best_path_states




states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}

observations_sequence = ['normal', 'cold', 'dizzy']

# E1: Given a set of observations X, calculate the occurrence probability of the observations X.
probability_X = hmm_probability(observations_sequence, states, start_probability, transition_probability, emission_probability)
print(f"Probability of observing the sequence: {probability_X}")

# E2: find the optimal sequence of hidden states
optimal_states = viterbi(observations_sequence, states, start_probability, transition_probability, emission_probability)
print(f"Optimal sequence of hidden states: {optimal_states}")




