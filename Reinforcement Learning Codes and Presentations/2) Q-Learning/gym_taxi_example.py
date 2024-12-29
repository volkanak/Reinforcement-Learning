import gym
import time

# Create and initialize the environment
env = gym.make("Taxi-v3", render_mode="ansi")
initial_state = env.reset()

# Now we can render safely
env.render()

# Print environment information
print("State space:", env.observation_space)  # 500 possible states
print("Action space:", env.action_space)      # 6 possible actions

"""
Environment details:
- Blue: passenger location
- Purple: destination
- Yellow/Red: empty taxi
- Green: full taxi
- RGBY: possible locations for pickup/dropoff
"""

# Demonstrate state encoding
taxi_row, taxi_col = 3, 1
passenger_loc = 2    # Index of passenger location
destination = 3      # Index of destination
state = env.encode(taxi_row, taxi_col, passenger_loc, destination)
print(f"Encoded state: {state}")

env.s = state
env.render()

"""
Available Actions:
0: move south
1: move north
2: move east 
3: move west 
4: pickup passenger
5: dropoff passenger
"""

# Run episodes
total_reward_list = []
num_episodes = 5

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    list_visualize = []
    done = False
    timestep = 0
    
    while not done:
        timestep += 1
        
        # Choose random action
        action = env.action_space.sample()
        
        # Take action - note we're now handling 5 return values
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine both termination conditions
        
        # Accumulate reward
        total_reward += reward
        
        # Store frame information
        list_visualize.append({
            "frame": env.render(),
            "state": next_state,
            "action": action,
            "reward": reward,
            "total_reward": total_reward
        })
        
        state = next_state
        
    total_reward_list.append(total_reward)
    print(f"Episode {episode + 1} completed with total reward: {total_reward}")

# Visualize episodes
for i, frame in enumerate(list_visualize):
    print("\nTimestep:", i + 1)
    print(frame["frame"])
    print("State:", frame["state"])
    print("Action:", frame["action"])
    print("Reward:", frame["reward"])
    print("Total Reward:", frame["total_reward"])
    time.sleep(0.5)

print("\nRewards across episodes:", total_reward_list)
