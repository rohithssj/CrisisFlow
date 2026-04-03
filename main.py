import os
import yaml
import argparse
from crisisflow.agents.base_agent import BaseAgent
from crisisflow.environment import CrisisEnv

def train(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize environment with CrisisEnv
    env = CrisisEnv(difficulty="medium")

    # Initialize agent
    agent = BaseAgent(env, config['agent_params'])

    # Start training
    print("Training CrisisFlow...")
    agent.train(episodes=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrisisFlow Trainer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    
    train(args.config)
