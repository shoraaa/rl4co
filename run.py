import os
import json
import torch
import pprint
import numpy as np
import random
import warnings
from options import get_options

import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import SSPkoptEnv
from rl4co.models import NeuOpt
from rl4co.utils.trainer import RL4COTrainer

import torch.nn as nn

class CustomizeATSPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, num_loc, linear_bias=True):
        super(CustomizeATSPInitEmbedding, self).__init__()
        node_dim = num_loc
        self.init_embed = nn.Sequential(
            # nn.LayerNorm(node_dim),
            nn.Linear(node_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )


    def forward(self, td):
        out = self.init_embed(td["cost_matrix"])
        return out
    
class CustomizeSSPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, fixed_len, linear_bias=True):
        super(CustomizeSSPInitEmbedding, self).__init__()
        node_dim = fixed_len
        self.init_embed = nn.Sequential(
            nn.Linear(node_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, td):
        out = self.init_embed(td["codes"])
        return out

class CustomizeSVDInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        super(CustomizeSVDInitEmbedding, self).__init__()
        node_dim = 2
        self.init_embed = nn.Sequential(
            # nn.LayerNorm(node_dim),
            nn.Linear(node_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )


    def forward(self, td):
        out = self.init_embed(td["locs_mds"])
        return out
    

def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SSPkoptEnv(generator_params=dict(num_loc=opts.graph_size, fixed_len=opts.string_length, init_sol_type=opts.init_val_met), k_max=opts.k)
    embedding = CustomizeATSPInitEmbedding(embed_dim=opts.embedding_dim, num_loc=opts.graph_size) if opts.embedding_type == "cost" else \
                CustomizeSSPInitEmbedding(embed_dim=opts.embedding_dim, fixed_len=opts.string_length) if opts.embedding_type == "codes" else \
                CustomizeSVDInitEmbedding(embed_dim=opts.embedding_dim)
    model = NeuOpt(
            env,

            ppo_epochs=opts.K_epochs,
            clip_range=opts.eps_clip,
            T_train=opts.T_train,
            n_step=opts.n_step,

            train_data_size=opts.epoch_size,
            val_data_size=opts.val_size,
            test_data_size=opts.test_size,

            batch_size=opts.batch_size,
            val_batch_size=opts.val_batch_size,
            test_batch_size=opts.test_batch_size,

            T_test=opts.T_max,
            CL_best=True,
            gamma=opts.gamma,
            lr_policy=opts.lr_model,
            lr_critic=opts.lr_critic,
            lr_scheduler_kwargs=dict(
                gamma=opts.lr_decay,
            ),

            CL_scalar=opts.warm_up,
            max_grad_norm=opts.max_grad_norm,

            policy_kwargs=dict(
                embed_dim=opts.embedding_dim,
                num_encoder_layers=opts.n_encode_layers,
                num_heads=opts.actor_head_num,
                feedforward_hidden=opts.hidden_dim,
                normalization=opts.normalization,
                tanh_clipping=opts.v_range,
                init_embedding=embedding,
            ),

            critic_kwargs=dict(
                embed_dim=opts.embedding_dim,
                num_heads=opts.critic_head_num,
                feedforward_hidden=opts.hidden_dim,
                normalization=opts.normalization,
            ),

            metrics=dict(
                train=["loss", "surrogate_loss", "value_loss", "cost_bsf", "cost_init"],
                val=["cost_bsf", "cost_init"],
                test=["cost_bsf", "cost_init"],
            ),

        )
    checkpoint_callback = ModelCheckpoint(dirpath=opts.save_dir, 
                                        filename="neuopt_epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/cost_bsf", # monitor validation reward
                                        mode="max") # maximize validation reward


    rich_model_summary = RichModelSummary(max_depth=3)
    callbacks = [checkpoint_callback, rich_model_summary]

    # Logger
    logger = WandbLogger(project="ssp_neuopt", name=opts.run_name) if not opts.no_wb else None
    
    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        model = model.load_from_checkpoint(load_path, strict=False)

    trainer = RL4COTrainer(
        max_epochs=opts.epoch_end,
        gradient_clip_val=opts.max_grad_norm,
        devices=1,
        accelerator="gpu",
        logger=logger,
        callbacks=callbacks,
        # log_every_n_steps=opts.log_every_n_steps,
    )

    # Do validation only
    if opts.eval_only:
        trainer.validate(model)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
    
        # Start the actual training loop
        trainer.fit(model)


if __name__ == "__main__":
    run(get_options())
