import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from snake_game import SnakeGameAI
from agent import Agent
agent=Agent(); agent.model.load_state_dict(torch.load("./checkpoints/model.pth")); agent.model.eval(); agent.epsilon=-100
game=SnakeGameAI()
while True:
    state=agent.get_state(game)
    # Epsilon 0 -> sirf model ki suno, random nahi
    state0=torch.tensor(state, dtype=torch.float); move=torch.argmax(agent.model(state0)).item()
    final=[0,0,0]; final[move]=1
    reward, done, score=game.play_step(final)
    if done:
        print("Game Over, Score:", score); game.reset()
