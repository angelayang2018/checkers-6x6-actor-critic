from mycheckersenv import Checkers6x6Env
from myagent import ACAgent, encode_action, decode_action 
import torch

# =========================
# Training Loop (Self-Play)
# =========================

def train(episodes=5000):
    env = Checkers6x6Env()
    agent = ACAgent()

    for episode in range(episodes):
        env.reset()

        log_probs = []
        values = []
        rewards = []

        while True:
            agent_name = env.agent_selection
            state = env.observe(agent_name)

            # Determine player perspective
            player = 1 if agent_name == "player_0" else -1

            # Get legal moves from environment
            legal_moves = env._get_legal_moves(player)

            # If no legal moves, pass dummy action
            if len(legal_moves) == 0:
                action = (0, 0, 0, 0)
                log_prob = torch.tensor(0.0)
                value = torch.tensor([[0.0]])
            else:
                action, log_prob, value = agent.act(state, legal_moves)

            env.step(action)

            reward = env.rewards[agent_name]

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            # End of episode
            if all(env.dones.values()):
                break

        loss = agent.update(log_probs, values, rewards)


        if episode % 100 == 0:
            print(f"Episode {episode} | Loss: {loss:.4f}")

    return agent


def print_board(board):
    print("\nBoard:")
    for row in board:
        print(" ".join(f"{int(x):2d}" for x in row))
    print()
    
def run_sample_game(agent):
    env = Checkers6x6Env()
    env.reset()

    print("\n===== SAMPLE GAME =====\n")
    print_board(env.board)

    step_count = 0

    while True:
        agent_name = env.agent_selection
        state = env.observe(agent_name)

        player = 1 if agent_name == "player_0" else -1
        legal_moves = env._get_legal_moves(player)

        if len(legal_moves) == 0:
            action = (0, 0, 0, 0)
        else:
            # Greedy action (no randomness for demo)
            with torch.no_grad():
                logits, _ = agent.model(torch.FloatTensor(state).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1).squeeze()

                mask = torch.zeros_like(probs)
                for move in legal_moves:
                    idx = encode_action(move)
                    mask[idx] = 1

                probs = probs * mask
                probs = probs / probs.sum()

                action_idx = torch.argmax(probs).item()
                action = decode_action(action_idx)

        env.step(action)

        step_count += 1
        print(f"Step {step_count} | {agent_name} -> {action}")
        print_board(env.board)

        if all(env.dones.values()):
            print("===== GAME OVER =====")
            print("Rewards:", env.rewards)
            break

# =========================
# Run Training
# =========================

if __name__ == "__main__":
    trained_agent = train(episodes=5000)
    run_sample_game(trained_agent)