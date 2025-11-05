"""Example: Run a local match between two agents.

This example shows how to run a match between two agents and save
a replay of the game.
"""

import rl_arena
from rl_arena.core.agent import RandomAgent
from rl_arena.utils import ReplayRecorder
from typing import List, Any


class SimpleHeuristicAgent:
    """
    A simple rule-based agent for Pong.

    Strategy: Move the paddle towards the ball's Y position.
    """

    def __init__(self, player_id: int):
        """
        Initialize the heuristic agent.

        Args:
            player_id: The player ID (0 or 1)
        """
        self.player_id = player_id

    def act(self, observation: Any) -> int:
        """
        Choose an action based on the ball position.

        Args:
            observation: Current observation array
                [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y, score1, score2]

        Returns:
            Action: 0 (UP), 1 (STAY), or 2 (DOWN)
        """
        ball_y = observation[1]
        my_paddle_y = observation[4] if self.player_id == 0 else observation[5]

        # Move towards the ball
        if ball_y < my_paddle_y - 0.02:  # Ball is above paddle
            return 0  # Move UP
        elif ball_y > my_paddle_y + 0.02:  # Ball is below paddle
            return 2  # Move DOWN
        else:
            return 1  # STAY


def run_match(
    env_name: str,
    agents: List[Any],
    agent_names: List[str],
    seed: int = None,
    render: bool = True,
    save_replay: bool = True,
    replay_path: str = "replays/match.json",
) -> dict:
    """
    Run a match between agents.

    Args:
        env_name: Name of the environment
        agents: List of agent objects
        agent_names: Names of the agents for display
        seed: Random seed
        render: Whether to render the game
        save_replay: Whether to save a replay
        replay_path: Path to save the replay

    Returns:
        Dictionary with match results
    """
    print("=" * 60)
    print(f"Match: {agent_names[0]} vs {agent_names[1]}")
    print("=" * 60)

    # Create environment
    env = rl_arena.make(env_name)

    # Setup replay recorder
    recorder = None
    if save_replay:
        recorder = ReplayRecorder(
            environment_name=env_name,
            configuration=env.get_config(),
            metadata={
                "player_names": agent_names,
                "seed": seed,
            },
        )

    # Reset environment
    observations, info = env.reset(seed=seed)
    print(f"Starting match... (seed={seed})")

    # Run match
    done = False
    step = 0
    total_rewards = [0.0, 0.0]

    while not done:
        # Get actions from agents
        actions = [agent.act(obs) for agent, obs in zip(agents, observations)]

        # Step environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Track rewards
        for i, reward in enumerate(rewards):
            total_rewards[i] += reward

        # Record step
        if recorder:
            recorder.record_step(
                step=step,
                actions=actions,
                observations=observations,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        # Render
        if render and step % 10 == 0:  # Render every 10 steps to avoid spam
            env.render(mode="ansi")

        step += 1

    # Final render
    if render:
        env.render(mode="ansi")

    # Save replay
    if recorder and save_replay:
        recorder.save(replay_path)
        print(f"\nReplay saved to: {replay_path}")

    # Print results
    print("\n" + "=" * 60)
    print("Match Results")
    print("=" * 60)
    print(f"Duration: {step} steps")
    print(f"Final Scores: {info['scores']}")
    print(f"Total Rewards: {total_rewards}")

    # Determine winner
    if info["scores"][0] > info["scores"][1]:
        winner = agent_names[0]
        winner_id = 0
    elif info["scores"][1] > info["scores"][0]:
        winner = agent_names[1]
        winner_id = 1
    else:
        winner = "Draw"
        winner_id = None

    print(f"Winner: {winner}")
    print("=" * 60)

    env.close()

    return {
        "winner": winner,
        "winner_id": winner_id,
        "scores": info["scores"],
        "total_rewards": total_rewards,
        "steps": step,
        "agent_names": agent_names,
    }


def main():
    """Run example matches."""
    import argparse

    parser = argparse.ArgumentParser(description="Run local matches")
    parser.add_argument("--env", type=str, default="pong", help="Environment name (default: pong)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-replay", action="store_true", help="Don't save replay")

    args = parser.parse_args()

    # Example 1: Random vs Random
    print("\nMatch 1: Random vs Random")
    env = rl_arena.make(args.env)
    agents = [
        RandomAgent(env.action_space, player_id=0),
        RandomAgent(env.action_space, player_id=1),
    ]
    run_match(
        env_name=args.env,
        agents=agents,
        agent_names=["Random Agent 1", "Random Agent 2"],
        seed=args.seed,
        render=not args.no_render,
        save_replay=not args.no_replay,
        replay_path="replays/random_vs_random.json",
    )

    # Example 2: Heuristic vs Random
    print("\n\nMatch 2: Heuristic vs Random")
    agents = [
        SimpleHeuristicAgent(player_id=0),
        RandomAgent(env.action_space, player_id=1),
    ]
    run_match(
        env_name=args.env,
        agents=agents,
        agent_names=["Heuristic Agent", "Random Agent"],
        seed=args.seed,
        render=not args.no_render,
        save_replay=not args.no_replay,
        replay_path="replays/heuristic_vs_random.json",
    )

    # Example 3: Heuristic vs Heuristic
    print("\n\nMatch 3: Heuristic vs Heuristic")
    agents = [
        SimpleHeuristicAgent(player_id=0),
        SimpleHeuristicAgent(player_id=1),
    ]
    run_match(
        env_name=args.env,
        agents=agents,
        agent_names=["Heuristic Agent 1", "Heuristic Agent 2"],
        seed=args.seed,
        render=not args.no_render,
        save_replay=not args.no_replay,
        replay_path="replays/heuristic_vs_heuristic.json",
    )


if __name__ == "__main__":
    main()
