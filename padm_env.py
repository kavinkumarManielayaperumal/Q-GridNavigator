import gymnasium as gym
import numpy as np
import pygame

class CustomEnv(gym.Env):
    """
    Custom Environment that follows the OpenAI gym interface.
    This environment simulates a grid with restricted areas and specific goal/hell states.
    This environment is based on One Piece anime. The agent state represents the hero crew,
    the hell states represent pirates and navy, and the goal state represents the treasure.
    """
    def __init__(self, grid_size=(11,10), accessible_area=(9, 8), 
                 background_image_path=None, agent_image_path=None, hell_images_paths=None, goal_image_path=None, 
                 hell_states=None, wall_states=None, boundary_states=None, goal_state=None, invisible_reward_coords=(5, 5)):
        super().__init__()
        self.grid_size = grid_size
        self.accessible_area = accessible_area
        
        # Calculate offsets to center the accessible area
        self.x_offset = (grid_size[0] - accessible_area[0]) // 2
        self.y_offset = (grid_size[1] - accessible_area[1]) // 2
        
        # Initial and goal states
        self.agent_state = np.array([self.x_offset, self.y_offset + self.accessible_area[1] - 1])
        self.goal_state = goal_state if goal_state is not None else np.array([self.x_offset + self.accessible_area[0] - 1, self.y_offset + self.accessible_area[1] - 1])
        
        # Restricted column and allowed point within it
        self.restricted_column = 4 + self.x_offset
        self.accessible_point = np.array([self.restricted_column, self.y_offset + 4])

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32)

        # Load and scale images
        self.cell_size = 70  # Reduced cell size for a smaller window
        self.background_image = pygame.transform.scale(pygame.image.load(background_image_path), (self.cell_size * grid_size[1], self.cell_size * grid_size[0]))
        self.agent_image = pygame.transform.scale(pygame.image.load(agent_image_path), (self.cell_size, self.cell_size))
        self.hell_images = [pygame.transform.scale(pygame.image.load(path), (self.cell_size, self.cell_size)) for path in hell_images_paths]
        self.goal_image = pygame.transform.scale(pygame.image.load(goal_image_path), (self.cell_size, self.cell_size))

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size * grid_size[1], self.cell_size * grid_size[0]))
        pygame.display.set_caption("Custom Environment")
        self.clock = pygame.time.Clock()

        # Initialize other attributes
        self.hell_states = hell_states if hell_states else []
        self.wall_states = wall_states if wall_states else []
        self.boundary_states = boundary_states if boundary_states else []
        self.invisible_reward_coords = invisible_reward_coords
        self.invisible_reward = False

    def step(self, action):
        # Define movement actions
        moves = {
            0: np.array([0, 1]),   # Right
            1: np.array([0, -1]),  # Left
            2: np.array([-1, 0]),  # Up
            3: np.array([1, 0])    # Down
        }

        # Compute the new state
        new_state = self.agent_state + moves[action]

        # Check if the new state is within the accessible area and not a wall or boundary
        if (
            self.y_offset <= new_state[0] < self.y_offset + self.accessible_area[0] and
            self.x_offset <= new_state[1] < self.x_offset + self.accessible_area[1] and
            tuple(new_state) not in self.wall_states and
            tuple(new_state) not in self.boundary_states
        ):
            self.agent_state = new_state

        # Compute the reward
        reward = -1  # Default reward
        done = False

        # Check if the agent reached the goal state
        if np.array_equal(self.agent_state, self.goal_state):
            reward = 20
            done = True

        # Check if the agent reached a hell state
        if tuple(self.agent_state) in self.hell_states:
            reward = -10
            done = True

        # Check if the agent reached the restricted column
        if self.agent_state[1] == self.restricted_column and not np.array_equal(self.agent_state, self.accessible_point):
            reward = -10
            done = True

        # Check if the agent reached the invisible reward point
        if np.array_equal(self.agent_state, self.invisible_reward_coords) and not self.invisible_reward:
            reward = 5
            self.invisible_reward = True

        # Return the new state, reward, and done flag
        return self.agent_state, reward, done, {}

    def reset(self):
        self.agent_state = np.array([self.x_offset, self.y_offset + self.accessible_area[1] - 1])
        self.invisible_reward = False
        return self.agent_state

    def render(self, mode='human'):
        self.screen.blit(self.background_image, (0, 0))
        
      

        # Draw hell states
        for i, state in enumerate(self.hell_states):
            self.screen.blit(self.hell_images[i % len(self.hell_images)], (state[1] * self.cell_size, state[0] * self.cell_size))

        # Draw the goal state
        self.screen.blit(self.goal_image, (self.goal_state[1] * self.cell_size, self.goal_state[0] * self.cell_size))

        # Draw the agent state
        self.screen.blit(self.agent_image, (self.agent_state[1] * self.cell_size, self.agent_state[0] * self.cell_size))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
