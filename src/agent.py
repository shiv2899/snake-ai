import torch, random, numpy as np
from collections import deque
from model import Linear_QNet
MAX_MEMORY=100_000; BATCH_SIZE=1000; LR=0.001
class Agent:
    def __init__(self):
        self.n_games=0; self.epsilon=0; self.gamma=0.9; self.memory=deque(maxlen=MAX_MEMORY)
        self.model=Linear_QNet(11, 256, 3); self.trainer=QTrainer(self.model, lr=LR, gamma=self.gamma)
    def get_state(self, game):
        head=game.snake[0]
        point_l=game.Point(head.x-20, head.y); point_r=game.Point(head.x+20, head.y)
        point_u=game.Point(head.x, head.y-20); point_d=game.Point(head.x, head.y+20)
        dir_l=game.direction.name=="LEFT"; dir_r=game.direction.name=="RIGHT"
        dir_u=game.direction.name=="UP"; dir_d=game.direction.name=="DOWN"
        state=[
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < head.x, game.food.x > head.x, game.food.y < head.y, game.food.y > head.y
        ]
        return np.array(state, dtype=int)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE: mini_sample=random.sample(self.memory, BATCH_SIZE)
        else: mini_sample=self.memory
        states, actions, rewards, next_states, dones=zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    def get_action(self, state):
        self.epsilon=80-self.n_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.epsilon:
            move=random.randint(0,2); final_move[move]=1
        else:
            state0=torch.tensor(state, dtype=torch.float)
            prediction=self.model(state0); move=torch.argmax(prediction).item(); final_move[move]=1
        return final_move

import torch, torch.nn as nn, torch.optim as optim
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model=model; self.lr=lr; self.gamma=gamma; self.optimizer=optim.Adam(model.parameters(), lr=self.lr); self.criterion=nn.MSELoss()
    def train_step(self, state, action, reward, next_state, done):
        state=torch.tensor(np.array(state), dtype=torch.float); next_state=torch.tensor(np.array(next_state), dtype=torch.float)
        action=torch.tensor(np.array(action), dtype=torch.long); reward=torch.tensor(np.array(reward), dtype=torch.float)
        if len(state.shape)==1:
            state=torch.unsqueeze(state,0); next_state=torch.unsqueeze(next_state,0); action=torch.unsqueeze(action,0); reward=torch.unsqueeze(reward,0); done=(done,)
        pred=self.model(state); target=pred.clone()
        for idx in range(len(done)):
            Q_new=reward[idx]
            if not done[idx]: Q_new=reward[idx]+self.gamma*torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()]=Q_new
        self.optimizer.zero_grad(); loss=self.criterion(target, pred); loss.backward(); self.optimizer.step()
