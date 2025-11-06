"""
Simple example demonstrating the core library workflow.

This shows how rl-arena-env achieves its goals:
1. ✅ Easy environment setup
2. ✅ Agent training
3. ✅ Testing
4. ✅ Submission preparation
"""

import rl_arena

print("=" * 60)
print("RL-Arena Library Workflow Demo")
print("=" * 60)
print()

# ============================================================
# 1. Environment Setup (Easy!)
# ============================================================
print("1️⃣ Creating environment...")
env = rl_arena.make("pong")
print(f"   ✅ Environment created: {env}")
print()

# ============================================================
# 2. Using Built-in Agents (Easy testing!)
# ============================================================
print("2️⃣ Testing with built-in agents...")
random_agent = rl_arena.RandomAgent()
rule_agent = rl_arena.RuleBasedAgent()

# Quick test
observations, _ = env.reset()
action1 = random_agent.act(observations[0])
action2 = rule_agent.act(observations[1])
print(f"   Random agent action: {action1}")
print(f"   Rule-based agent action: {action2}")
print(f"   ✅ Agents working!")
print()

# ============================================================
# 3. Training (if stable-baselines3 is installed)
# ============================================================
print("3️⃣ Training agent...")
try:
    # Quick training (just 1000 steps for demo)
    print("   Training DQN for 1000 steps (demo)...")
    model = rl_arena.train_dqn(
        env_name="pong",
        total_timesteps=1000,
        verbose=0,
    )
    print("   ✅ Training completed!")
    
    # Save model
    model.save("demo_agent.zip")
    print("   ✅ Model saved: demo_agent.zip")
    print()
    
    # ============================================================
    # 4. Evaluation
    # ============================================================
    print("4️⃣ Evaluating agent...")
    agent = rl_arena.create_agent(model)
    results = rl_arena.evaluate(
        agent=agent,
        env_name="pong",
        n_episodes=5,
        render=False,
    )
    
    print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Mean length: {results['mean_length']:.2f}")
    print("   ✅ Evaluation completed!")
    print()
    
    # ============================================================
    # 5. Submission Preparation
    # ============================================================
    print("5️⃣ Creating submission file...")
    rl_arena.create_submission(
        agent=agent,
        output_path="demo_submission.py",
        agent_name="DemoAgent",
        description="Quick demo agent",
        author="demo_user",
        version="0.1.0",
    )
    print()
    
    # ============================================================
    # 6. Validation
    # ============================================================
    print("6️⃣ Validating submission...")
    validation = rl_arena.validate("demo_submission.py")
    
    if validation['valid']:
        print("   ✅ Submission is valid!")
        print(f"   File size: {validation['file_size'] / 1024:.1f} KB")
    else:
        print("   ❌ Issues found:")
        for issue in validation['issues']:
            print(f"      - {issue}")
    
    if validation['warnings']:
        print("   ⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
    
    print()
    
except ImportError:
    print("   ⚠️ Stable-Baselines3 not installed")
    print("   Install with: pip install stable-baselines3")
    print("   Skipping training demo...")
    print()
    
    # Show how to use pre-trained models
    print("   Alternative: Use built-in agents")
    print()
    print("   Example:")
    print("   >>> agent = rl_arena.RandomAgent()")
    print("   >>> rl_arena.create_submission(agent, 'submission.py')")
    print()

# ============================================================
# 7. Interactive Testing (optional)
# ============================================================
print("7️⃣ Interactive testing available!")
print("   Run: python -m rl_arena.interactive --player1 human --player2 random")
print("   Or: python examples/play_interactive.py")
print()

print("=" * 60)
print("✅ Demo Complete!")
print("=" * 60)
print()
print("Summary of what we did:")
print("1. ✅ Created environment (1 line)")
print("2. ✅ Trained agent (1 line)")
print("3. ✅ Evaluated agent (1 line)")
print("4. ✅ Created submission (1 line)")
print("5. ✅ Validated submission (1 line)")
print()
print("Total: 5 lines of code for complete workflow! 🎉")
print()
print("Next steps:")
print("- Train longer: model = rl_arena.train_dqn('pong', total_timesteps=50000)")
print("- Test interactively: python -m rl_arena.interactive")
print("- Submit to backend: Upload demo_submission.py")
print()
