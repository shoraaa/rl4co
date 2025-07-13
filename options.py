import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(description="NeuOpt-GIRE")

    ### overall settings
    parser.add_argument('--graph_size', type=int, default=10, help="The size of the problem graph")
    parser.add_argument('--string_length', type=int, default=15, help="The length of the string to be generated")
    parser.add_argument('--eval_only', action='store_true', help='used only if to evaluate a model')
    parser.add_argument('--init_val_met', choices = ['greedy', 'random'], default = 'random', help='method to generate initial solutions while validation')
    parser.add_argument('--embedding_type', choices=['codes', 'cost', 'svd'], default='codes', help='Type of embedding to use for the model')
    parser.add_argument('--no_wb', action='store_true', help='Disable Weight and Biases logging')
    # parser.add_argument('--log_every_n_steps', type=int, default=1, help='Log every n steps')
    parser.add_argument('--no_saving', action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--seed', type=int, default=6666, help='Random seed to use')
    
    ### NeuOpt configs
    parser.add_argument('--k', type=int, default=4) # the maximum basis move number K
    
    ### resume and load models
    parser.add_argument('--load_path', default = None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', default = None, help='Resume from previous checkpoint file')
    
    ### training AND validation
    parser.add_argument('--K_epochs', type=int, default=3, help='K mini-epochs per epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip parameter')
    parser.add_argument('--T_train', type=int, default=200, help='Number of training steps per epoch')
    parser.add_argument('--T_max', type=int, default=1000, help='maximum inference steps')
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances per batch during training')
    parser.add_argument('--epoch_end', type=int, default=20, help='End at epoch #')
    parser.add_argument('--epoch_size', type=int, default=1000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=100, help='Number of instances used for reporting validation performance')
    parser.add_argument('--test_size', type=int, default=100, help='Number of instances used for testing performance')
    parser.add_argument('--lr_model', type=float, default=8e-5, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=2e-5, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.985, help='Learning rate decay per epoch')
    parser.add_argument('--warm_up', type=float, default=2.0) # the rho in the paper
    parser.add_argument('--max_grad_norm', type=float, default=0.05, help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    
    ### network
    parser.add_argument('--v_range', type=float, default=6.)
    parser.add_argument('--critic_head_num', type=int, default=4)
    parser.add_argument('--actor_head_num', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--normalization', default='layer', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--gamma', type=float, default=0.999, help='decrease future reward')
    

    ### outputs
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--run_name', default='run_name', help='Name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')

    opts = parser.parse_args(args)
    
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format("ssp", opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None
    return opts