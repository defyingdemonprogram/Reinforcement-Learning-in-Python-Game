import torch
import random
import numpy as np
from collections import deque
from env import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self, model_path: str = None):
        self.n_games = 0
        # Do exploration in first 100 game if model start from scratch
        self.exploration = 5
        self.epsilon = 0  # Randomness factor
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Stores experiences
        self.model = LinearQNet(11, 256, 3)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    def get_state(self, game) -> np.ndarray:
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        directions = {
            'left': game.direction == Direction.LEFT,
            'right': game.direction == Direction.RIGHT,
            'up': game.direction == Direction.UP,
            'down': game.direction == Direction.DOWN
        }
        directions["left"] = game.direction == Direction.LEFT
        directions["right"] = game.direction == Direction.RIGHT
        directions["up"] = game.direction == Direction.UP
        directions["down"] = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (directions["right"] and game.is_collision(point_r)) or 
            (directions["left"] and game.is_collision(point_l)) or 
            (directions["up"] and game.is_collision(point_u)) or 
            (directions["down"] and game.is_collision(point_d)),

            # Danger right
            (directions["up"] and game.is_collision(point_r)) or 
            (directions["down"] and game.is_collision(point_l)) or 
            (directions["left"] and game.is_collision(point_u)) or 
            (directions["right"] and game.is_collision(point_d)),

            # Danger left
            (directions["down"] and game.is_collision(point_r)) or 
            (directions["up"] and game.is_collision(point_l)) or 
            (directions["right"] and game.is_collision(point_u)) or 
            (directions["left"] and game.is_collision(point_d)),
            
            # Move direction
            directions["left"],
            directions["right"],
            directions["up"],
            directions["down"],
            
            # Food location 
            game.food.x < game.head.x,  # food is to the left
            game.food.x > game.head.x,  # food is to the right
            game.food.y < game.head.y,  # food is above
            game.food.y > game.head.y,  # food is below
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # Decay epsilon over time
        self.epsilon = self.exploration - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(model_path: str = None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # Make record 0 if training from scratch
    record = 30

    agent = Agent(model_path)
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game {agent.n_games}, Score {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    model_path = "./model/model.pth"
    train(model_path=model_path)
