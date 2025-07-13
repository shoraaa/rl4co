"""
Example script demonstrating how to use DeepACO with PPO algorithm.
"""

import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.deepaco import DeepACOPPO
from rl4co.utils.trainer import RL4COTrainer

def main():
    # Create environment
    env = TSPEnv(num_loc=20)
    
    # Create DeepACO model with PPO
    model = DeepACOPPO(
        env=env,
        train_with_local_search=True,
        ls_reward_aug_W=0.95,
        # PPO specific parameters
        clip_range=0.2,
        ppo_epochs=4,
        mini_batch_size=0.25,
        vf_lambda=0.5,
        entropy_lambda=0.01,
        normalize_adv=True,
        max_grad_norm=0.5,
        # Policy parameters
        policy_kwargs={
            "n_ants": {"train": 30, "val": 48, "test": 48},
            "n_iterations": {"train": 1, "val": 5, "test": 10},
        },
    )
    
    # Create trainer
    trainer = RL4COTrainer(
        max_epochs=10,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
    )
    
    # Train the model
    trainer.fit(model)
    
    # Test the model
    test_data = env.dataset(batch_size=[2])
    td = env.reset(test_data)
    
    with torch.no_grad():
        out = model.policy(td, env, phase="test")
        print(f"Test rewards: {out['reward']}")
        print(f"Test actions shape: {out['actions'].shape}")

if __name__ == "__main__":
    main()