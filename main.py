from padm_env import CustomEnv
from q_learning import train_q_learning, visualize_q_table

if __name__ == "__main__":
    # Define file paths for images
    background_image_path = r'background.jpeg'
    agent_image_path = r'agent image.png'
    goal_image_path = r'goal state.jpeg'
    hell_images_paths = [
        r'hell state 1.png',
        r'hell state 2.png',
        r'hell state 3.png',
        r'hell state 4.png',
        r'hell state 5.png',
        r'hell state 6.png'
    ]
    
    # Define hell state coordinates
    hell_states = [
        (6, 7),
        (3, 8),
        (5, 2),
        (1, 7),
        (7, 2),
        (2, 3)
    ]

    # Define wall coordinates
    wall_states = [
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (6, 5),
        (7, 5),
        (8, 5),
        (9, 5)
    ]

    # Define boundary coordinates
    boundary_states = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
        (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10),
        (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)
    ]
    
    # Define goal state coordinates
    goal_state = (8, 1)
    
    # Create the custom environment
    env = CustomEnv(grid_size=(11,10), accessible_area=(9, 8), 
                    background_image_path=background_image_path, 
                    agent_image_path=agent_image_path, 
                    hell_images_paths=hell_images_paths,
                    goal_image_path=goal_image_path,
                    hell_states=hell_states,
                    wall_states=wall_states,
                    boundary_states=boundary_states,
                    goal_state=goal_state)
    
    # Train the Q-learning agent
    q_table = train_q_learning(env, no_episodes=500, epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995, alpha=0.1, gamma=0.99)
    
    # Visualize the Q-table
    visualize_q_table(env)
    
    # Close the environment
    env.close()
