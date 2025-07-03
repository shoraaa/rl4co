#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

from rl4co.envs import SSPkoptEnv, ATSPkoptEnv, TSPkoptEnv
from rl4co.models import NeuOptPolicy, NeuOpt
from rl4co.models.nn.env_embeddings.edge import ATSPEdgeEmbedding
from rl4co.utils.trainer import RL4COTrainer


# In[ ]:


import torch.nn as nn
from rl4co.envs.routing.atsp.generator import ATSPCoordGenerator

class CustomizeATSPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, num_loc, linear_bias=True):
        super(CustomizeATSPInitEmbedding, self).__init__()
        node_dim = 5
        self.init_embed = nn.Sequential(
            # nn.LayerNorm(node_dim),
            nn.Linear(node_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )


    def forward(self, td):
        out = self.init_embed(td["locs_mds"])
        return out

num_loc = 10  # Number of strings
fixed_len = 15
embed_dim = 128  # Dimension of the embedding space

env = SSPkoptEnv(generator_params=dict(num_loc=num_loc, fixed_len=fixed_len, init_sol_type="random"), k_max=4)
model = NeuOpt(
        env,
        batch_size=128,
        train_data_size=1000,
        val_data_size=100,
        test_data_size=100,
        n_step=5,
        T_train=200,
        T_test=1000,
        CL_best=True,
        policy_kwargs=dict(
            embed_dim=embed_dim,
            init_embedding=CustomizeATSPInitEmbedding(num_loc=num_loc,embed_dim=embed_dim),
        ),
    )


# In[3]:


checkpoint_callback = ModelCheckpoint(  dirpath="checkpoints_ssp", # save to checkpoints/
                                        filename="atsp_mds{epoch:03d}",  # save as epoch_XXX.ckpt
                                        save_top_k=1, # save only the best model
                                        save_last=True, # save the last model
                                        monitor="val/cost_bsf", # monitor validation reward
                                        mode="min") # maximize validation reward


rich_model_summary = RichModelSummary(max_depth=3)

callbacks = [checkpoint_callback, rich_model_summary]


# In[4]:


from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project="rl4co", name="atsp_mds_large", log_model=True, save_dir="wandb_logs")

#  logger = None


# In[5]:


trainer = RL4COTrainer(
    max_epochs=20,
    gradient_clip_val=0.05,
    devices=1,
    accelerator="gpu",
    logger=logger,
    callbacks=callbacks,
)


# In[6]:


# trainer.test(model)


# In[ ]:


# Fit and test the model
trainer.fit(model)
trainer.test(model)

