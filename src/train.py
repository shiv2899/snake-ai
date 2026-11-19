import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from snake_game import SnakeGameAI
from agent import Agent
import matplotlib.pyplot as plt

def train():
    scores=[]; mean_scores=[]; total_score=0; record=0
    agent=Agent(); game=SnakeGameAI()
    while True:
        state_old=agent.get_state(game)
        final_move=agent.get_action(state_old)
        reward, done, score=game.play_step(final_move)
        state_new=agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset(); agent.n_games+=1; agent.train_long_memory()
            if score>record:
                record=score; agent.model.save()
                print(f"New Record! Game {agent.n_games} Score {score}")
            print(f"Game {agent.n_games} Score {score} Record {record}")
            scores.append(score); total_score+=score; mean_score=total_score/agent.n_games; mean_scores.append(mean_score)
            plt.clf(); plt.title("Training..."); plt.xlabel("Games"); plt.ylabel("Score")
            plt.plot(scores, label="Score"); plt.plot(mean_scores, label="Mean"); plt.legend(); plt.pause(0.01)
            if agent.n_games>=200:
                print(f"Done! Best {record}"); break
if __name__=="__main__": train()
