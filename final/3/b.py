import gym
import numpy as np
import timeit

def print_policy(policy):
    #print policy in wanted format
    moves = ['L' ,'D' ,'R' ,'U']
    for i in range(5):
        for j in range(5):
            ind = 5*i + j
            print(moves[int(np.argmax(policy[ind]))] , end ='')
        print()
            
def policy_evaluation(policy, env, discount=1.0, theta=10**(-9), max_iter=10**9):
    """a function for calculating V for a policy

    Args:
        policy ([numpy array]): [for each state has probability of each action]
        env ([gym.env]): [frozen lake environment]
        discount (float, optional): [discount rate]. Defaults to 1.0.
        theta ([float], optional): [trashhold for change in values]. Defaults to 10**(-10).
        max_iter ([int], optional): [numbet of max iterations for algorithm]. Defaults to 10**9.

    Returns:
        [numpy array]: [state value function for given policy]
    """    
    
    evaluation_iterations = 1
    # set V0 = 0 for each state
    V = np.zeros(env.nS)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iter)):
        # Initialize a change of value function as zero
        delta = 0
        
        for state in range(env.nS):
            
            v = 0
            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(policy[state]):
                # Check how good next state will be
                for state_probability, next_state, reward, done in env.P[state][action]:
                    # Calculate the expected value
                    v += action_probability * state_probability * (reward + discount * V[next_state])
                       
            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))
            # Update value function
            V[state] = v
        evaluation_iterations += 1
                
        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V
                    
def policy_iteration(env, discount=.85, max_iter=10**9):
    """performing policy iteration algorithm

    Args:
        env ([gym.env]): [frozen lake environment]
        discount (float, optional): [discount_factor]. Defaults to 0.85.
        max_iter ([float], optional): [numbet of max iterations for algorithm]. Defaults to 1e9.

    Returns:
        policy [numpy array]: [best policy for each state]
        V [numpy array]: [state value function]
    """    
    #first policy is a random uniform policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    evaluated_policies = 1
    
    for i in range(int(max_iter)):
        stable_policy = True
        # Evaluate current policy
        V = policy_evaluation(policy, env, discount=discount)
        #policy Improvement
        for state in range(env.nS):
            # Choose the best action in a current state under current policy
            current_action = np.argmax(policy[state])
            # calculate values for all possible actions in this state
            action_value = next_step(env, state, V, discount)
            # Select best action
            best_action = np.argmax(action_value)
            # If action didn't change
            if current_action != best_action:
                stable_policy = False
                policy[state] = np.eye(env.nA)[best_action]
        evaluated_policies += 1

        if stable_policy:
            print(f'Evaluated {evaluated_policies} policies.')
            return policy, V
    
def next_step(env, state, V, discount):
    """calculate V in next step for a state for each action

    Args:
        env ([gym.env]): [frozen lake environment]
        state ([int]): [number of state ]
        V ([numpy array ]): [values matrix]
        discount_factor ([float]): [discount_factor]

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
policy, V =policy_iteration(env)
stop = timeit.default_timer()

np.set_printoptions(precision=9 ,suppress = True)
print('V :')
print(V.reshape(5,5))
print()
print('optimal policy:')
print_policy(policy)
print('Time: ', stop - start) 







 
