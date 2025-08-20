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

from rl4co.envs import SSPEnv
from rl4co.models import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

from rl4co.utils.ops import gather_by_index

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
    
class SSPContext(nn.Module):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embedding_dim,  linear_bias=True):
        super(SSPContext, self).__init__()
        self.W_placeholder = nn.Parameter(
            torch.Tensor(embedding_dim).uniform_(-1, 1)
        )
        self.project_context = nn.Linear(
            embedding_dim, embedding_dim, bias=linear_bias
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)
        
class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


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

    env = SSPEnv(generator_params=dict(num_loc=opts.graph_size, fixed_len=opts.string_length, init_sol_type=opts.init_val_met))
    embedding = CustomizeATSPInitEmbedding(embed_dim=opts.embedding_dim, num_loc=opts.graph_size) if opts.embedding_type == "cost" else \
                CustomizeSSPInitEmbedding(embed_dim=opts.embedding_dim, fixed_len=opts.string_length) if opts.embedding_type == "codes" else \
                CustomizeSVDInitEmbedding(embed_dim=opts.embedding_dim)
    model = AttentionModel(
            env,

            batch_size=opts.batch_size,
            train_data_size=opts.epoch_size,
            val_data_size=opts.val_size,
            test_data_size=opts.test_size,

            val_batch_size=opts.val_batch_size,
            test_batch_size=opts.test_batch_size,

            optimizer="Adam",
            optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            lr_scheduler=opts.lr_scheduler,
            lr_scheduler_kwargs={"milestones": [1901, ], "gamma": opts.lr_scheduler_gamma},

            policy_kwargs=dict(
                embed_dim=opts.embedding_dim,
                num_encoder_layers=opts.n_encode_layers,
                num_heads=opts.actor_head_num,
                feedforward_hidden=opts.hidden_dim,
                normalization=opts.normalization,
                tanh_clipping=opts.v_range,
                init_embedding=embedding,
                context_embedding=SSPContext(embedding_dim=opts.embedding_dim),
                dynamic_embedding=StaticEmbedding(opts.embedding_dim),
                use_graph_context=False
            ),

            metrics=dict(
                train=["loss", "value_loss, reward"],
                val=["reward"],
                test=["reward"],
            ),

        )
    checkpoint_callback = ModelCheckpoint(dirpath=opts.save_dir, 
                                        filename="am_epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor=opts.monitor, # monitor validation reward
                                        mode=opts.monitor_mode) # maximize validation reward


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
