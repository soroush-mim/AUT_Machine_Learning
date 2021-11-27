import gym
import numpy as np
import timeit

def print_policy(policy):
    #print policy in wanted format
    moves = ['L' ,'D' ,'R' ,'U']
    for i in range(5):
        for j in range(5):
            ind = 5*i + j
            print(moves[int(policy[ind])] , end ='')
        print()
                
def value_iteration(env, discount=0.85, theta=10**(-9), max_iter=10**9):
    """performing value iteration for 

    Args:
        env ([gym.env]): [frozen lake environment]
        discount (float, optional): [discount_factor]. Defaults to 0.85.
        theta ([float], optional): [trashhold for change in values]. Defaults to 1e-10.
        max_iter ([float], optional): [numbet of max iterations for algorithm]. Defaults to 1e9.

    Returns:
        policy [numpy array]: [best policy for each state]
        V [numpy array]: [state value function]
    """    
    # set V0 = 0 for all states
    V = np.zeros(env.nS)
    for i in range(max_iter):
        # Early stopping condition
        delta = 0
        # Update each state
        for state in range(env.nS):
            # calculate values for all possible actions in this state
            action_values = next_step(env, state, V, discount)
            # Select best action
            best_action_value = np.max(action_values)
            # Calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))
            # Update the value function for current state
            V[state] = best_action_value
            # Check if we can stop
        if delta < theta:
            print('num of iterations for Value-iteration algorithm: ' + str(i))
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS])
    for state in range(env.nS):
        # One step lookahead to find the best action for this state
        action_value = next_step(env, state, V, discount)
        # Update the policy to perform a better action at a current state
        policy[state] = np.argmax(action_value)
            
    return policy, V
    
def next_step(env, state, V, discount):
    """calculate V in next step for a state for each action

    Args:
        env ([gym.env]): [frozen lake environment]
        state ([int]): [number of state ]
        V ([numpy array ]): [values matrix]
        discount  ([float]): [discount rate]

    Returns:
        [numpy array]: [value for each action from state]
    """    
    values = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, done in env.P[state][action]:
            values[action] += prob * (reward + discount * V[next_state])
    return values

#creating and initializing the map
MAP = [ 'SFFFH',
        'FFHHF',
        'FFFFF',
        'HHFHF',
        'FFFFG']

env = gym.make('FrozenLake-v0', desc=MAP)
env.reset()

start = timeit.default_timer()
policy, V =value_iteration(env)
stop = timeit.default_timer()

print('V :')
print(V.reshape(5,5))
print()

print('optimal policy:')
print_policy(policy)
print('Time: ', stop - start) 







 
