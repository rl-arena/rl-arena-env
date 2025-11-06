"""
Interactive Play Examples for RL-Arena

Run different play modes:
1. Human vs Random AI
2. Human vs Trained Agent
3. Agent vs Agent (watch AI battle)
4. Human vs Human (local multiplayer)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_arena.interactive import InteractivePlayer, load_builtin_agent


def play_vs_random():
    """Play against random AI."""
    print("\n🎮 Human vs Random AI")
    print("Controls: W/S for up/down")
    
    player = InteractivePlayer(env_name="pong", fps=60)
    player.play(
        player1_agent=None,  # Human
        player2_agent=load_builtin_agent("random"),
    )


def play_vs_trained():
    """Play against trained agent."""
    print("\n🎮 Human vs Trained AI")
    print("Controls: W/S for up/down")
    print("Loading trained agent...")
    
    # Load your trained agent
    from stable_baselines3 import DQN
    from rl_arena.wrappers.gymnasium_wrapper import GymnasiumWrapper
    
    # Create wrapper for agent
    env = GymnasiumWrapper("pong", player_id=1)
    model = DQN.load("training_results/pong_dqn_agent.zip")
    
    class TrainedAgent:
        def act(self, observation, info=None):
            action, _ = model.predict(observation, deterministic=True)
            return int(action)
    
    player = InteractivePlayer(env_name="pong", fps=60)
    player.play(
        player1_agent=None,  # Human
        player2_agent=TrainedAgent(),
    )


def watch_ai_battle():
    """Watch AI agents battle."""
    print("\n🤖 AI vs AI Battle")
    print("Controls: SPACE=Pause, R=Reset, Q=Quit")
    
    player = InteractivePlayer(env_name="pong", fps=60)
    player.play(
        player1_agent=load_builtin_agent("rule_based"),
        player2_agent=load_builtin_agent("random"),
        max_episodes=5,
    )


def play_local_multiplayer():
    """Local 2-player mode."""
    print("\n👥 Local Multiplayer")
    print("Player 1: W/S")
    print("Player 2: Arrow Up/Down")
    
    player = InteractivePlayer(env_name="pong", fps=60)
    player.play(
        player1_agent=None,  # Human 1
        player2_agent=None,  # Human 2
    )


def main():
    """Show menu and run selected mode."""
    print("=" * 60)
    print("RL-ARENA INTERACTIVE PLAYER")
    print("=" * 60)
    print()
    print("Select play mode:")
    print("1. Human vs Random AI")
    print("2. Human vs Trained Agent (requires trained model)")
    print("3. Watch AI vs AI Battle")
    print("4. Local Multiplayer (Human vs Human)")
    print("5. Exit")
    print()
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        play_vs_random()
    elif choice == "2":
        try:
            play_vs_trained()
        except Exception as e:
            print(f"\n❌ Error loading trained agent: {e}")
            print("Please train an agent first using test_complete_workflow.py")
    elif choice == "3":
        watch_ai_battle()
    elif choice == "4":
        play_local_multiplayer()
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
