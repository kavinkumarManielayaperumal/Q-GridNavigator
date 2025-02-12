import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_q_learning(env, no_episodes=500, epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995, alpha=0.1, gamma=0.99, q_table_save_path="q_table.npy"):
    # Initialize Q-table with zeros
    q_table = np.zeros((*env.observation_space.high, env.action_space.n))
    
    for episode in range(no_episodes):
        state = env.reset()
        state = tuple(state)  # Convert state to tuple format for indexing
        total_reward = 0  # Initialize total reward for the episode
        done = False
        
        while not done:
            # Exploration vs. Exploitation trade-off
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore: Choose random action
            else:
                action = np.argmax(q_table[state])  # Exploit: Choose action with max Q-value
            
            next_state, reward, done, _ = env.step(action)
           # env.render()  # Render environment (optional)
            
            next_state = tuple(next_state)  # Convert next state to tuple format
            total_reward += reward  # Accumulate total reward for the episode
            
            # Q-table update using Q-learning formula
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            
            state = next_state  # Update current state to next state
        
        # Decay exploration rate epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    # Save learned Q-table
    np.save(q_table_save_path, q_table)
    print("Training finished and Q-table saved.")
    
def visualize_q_table(env, q_table_path="q_table.npy"):
    try:
        # Load the saved Q-table
        q_table = np.load(q_table_path)
        
        actions = ["Right", "Left", "Up", "Down"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False)
            ax.set_title(f'Action: {action}')
            
            # Annotate special states on each heatmap
            # Annotate goal state
            goal_state = env.goal_state
            ax.text(goal_state[1] + 0.5, goal_state[0] + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=8)
            
            # Annotate hell states
            for hell in env.hell_states:
                ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=8)
            
            # Annotate wall states
            for wall in env.wall_states:
                ax.text(wall[1] + 0.5, wall[0] + 0.5, 'W', color='black', ha='center', va='center', weight='bold', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
