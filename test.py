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

from rl4co.envs import SSPEnv, SSPkoptEnv
from rl4co.models import AttentionModel, NeuOpt
from rl4co.utils.trainer import RL4COTrainer

from rl4co.utils.ops import gather_by_index

import torch.nn as nn

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:45:10 2020

@author: Kenneth
"""

"""
Import modules needed.
"""
import itertools
import numpy as np
from tqdm import tqdm

"""
This function reads a FASTQ file.
"""
def readFastq(filename):
    sequences = []
    qualities = []
    with open(filename) as fh:
        while True:
            fh.readline()  # skip name line
            seq = fh.readline().rstrip()  # read base sequence
            fh.readline()  # skip placeholder line
            qual = fh.readline().rstrip() # base quality line
            if len(seq) == 0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities

"""
This function finds the length of the longest suffix of a which overlaps with a prefix of b.
"""
def overlap(a, b, min_length=3):
    """ Return length of longest suffix of 'a' matching
        a prefix of 'b' that is at least 'min_length'
        characters long.  If no such overlap exists,
        return 0. """
    start = 0  # start all the way at the left
    while True:
        start = a.find(b[:min_length], start)  # look for b's suffx in a
        if start == -1:  # no more occurrences to right
            return 0
        # found occurrence; check for full suffix/prefix match
        if b.startswith(a[start:]):
            return len(a)-start
        start += 1  # move just past previous match

"""
This function finds the set of shortest common superstrings of given strings.
Note that the given strings must have the same length.
"""
def scs(ss):
    """ Returns shortest common superstring of given
        strings, which must be the same length """
    shortest_sup = []
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss)-1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i+1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i+1][olen:]
        if len(shortest_sup) == 0 or len(sup) < len(shortest_sup[0]):
            shortest_sup = [sup]  # found shorter superstring
        elif len(sup) == len(shortest_sup[0]):
            shortest_sup.append(sup)
    return shortest_sup  # return shortest

"""
Given a set of reads, this function finds the pair which overlap the most and calculates the length of the overlap.
"""
def pick_maximal_overlap(reads, k):
    reada, readb = None, None
    best_olen = 0
    for a,b in itertools.permutations(reads, 2):
        olen = overlap(a, b, k)
        if olen > best_olen:
            reada, readb = a, b
            best_olen = olen
    return reada, readb, best_olen

"""
This function implements the greedy shortest common superstring algorithm.
"""
def greedy_scs(reads, k):
    read_a, read_b, olen = pick_maximal_overlap(reads, k)
    while olen > 0:
        # print(len(reads))
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        read_a, read_b, olen = pick_maximal_overlap(reads, k)
    return ''.join(reads)

"""
This is an accelerated version of pick_maximal_overlap(reads, k).
This is achieved by building an k-mer index so that not every permutation of reads is considered.
"""
def pick_maximal_overlap_index(reads, k):
    index = {}
    for read in reads:
        kmers = []
        for i in range(len(read) - k + 1):
            kmers.append(read[i:i+k])
        for kmer in kmers:
            if kmer not in index:
                index[kmer] = set()
            index[kmer].add(read)
    for read in reads:
        for i in range(len(read)-k+1):
            dummy = read[i:i+k]
            if dummy not in index:
                index[dummy] = set()
            index[dummy].add(read)
    reada, readb = None, None
    best_olen = 0
    for a in reads:
        for b in index[a[-k:]]:
            if a != b:
                olen = overlap(a, b, k)
                if olen > best_olen:
                    reada, readb = a, b
                    best_olen = olen
    return reada, readb, best_olen

"""
This function implements the greedy shortest common superstring algorithm using an accelerated version of pick_maximal_overlap(reads, k).
"""
def greedy_scs_index(reads, k):
    read_a, read_b, olen = pick_maximal_overlap_index(reads, k)
    while olen > 0:
        # print(len(reads))
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        read_a, read_b, olen = pick_maximal_overlap_index(reads, k)
    return ''.join(reads)      

from tensordict import TensorDict

def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    batch_size = x[list(x.keys())[0]].shape[0]
    return TensorDict(x, batch_size=batch_size)

class CustomizeATSPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, num_loc, linear_bias=True):
        super(CustomizeATSPInitEmbedding, self).__init__()
        node_dim = num_loc
        self.init_embed = nn.Sequential(
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
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:
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


def load_attention_model(checkpoint_path, opts):
    """Load AttentionModel from checkpoint"""
    env = SSPEnv(generator_params=dict(num_loc=opts.graph_size, fixed_len=opts.string_length, init_sol_type=opts.init_val_met), test_file=opts.test_file)
    embedding = CustomizeATSPInitEmbedding(embed_dim=opts.embedding_dim, num_loc=opts.graph_size) if opts.embedding_type == "cost" else \
                CustomizeSSPInitEmbedding(embed_dim=opts.embedding_dim, fixed_len=opts.string_length) if opts.embedding_type == "codes" else \
                CustomizeSVDInitEmbedding(embed_dim=opts.embedding_dim)
    
    model = AttentionModel.load_from_checkpoint(
        checkpoint_path,
        env=env,
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
        strict=False
    )
    return model


def load_neuopt_model(checkpoint_path, opts):
    """Load NeuOpt model from checkpoint"""
    env = SSPkoptEnv(generator_params=dict(num_loc=opts.graph_size, fixed_len=opts.string_length, init_sol_type=opts.init_val_met), k_max=opts.k, test_file=opts.test_file)
    embedding = CustomizeATSPInitEmbedding(embed_dim=opts.embedding_dim, num_loc=opts.graph_size) if opts.embedding_type == "cost" else \
                CustomizeSSPInitEmbedding(embed_dim=opts.embedding_dim, fixed_len=opts.string_length) if opts.embedding_type == "codes" else \
                CustomizeSVDInitEmbedding(embed_dim=opts.embedding_dim)
    
    model = NeuOpt.load_from_checkpoint(
        checkpoint_path,
        env=env,
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
        strict=False
    )
    return model

import time
from tqdm import tqdm 
def test_greedy(opts, k=2):
    codes = np.load('./data/data_ssp.npz')["codes"]
    num_ins = codes.shape[0]
    instance = []

    greedy_start = time.time()
    for i in range(num_ins):
        ssp = codes[i].astype(int).tolist()
        ss = []
        for j in ssp:
            ss.append("".join(str(x) for x in j))
        instance.append(ss)

    ans_greedy = []
    for i in range(num_ins):
        gr_ssstr = greedy_scs(instance[i], k)
        ans_greedy.append(len(gr_ssstr))
    ans_greedy = np.array(ans_greedy)
    greedy_time = time.time() - greedy_start

    return ans_greedy.mean(), greedy_time

def test_models(opts):
    """Test and compare AttentionModel and NeuOpt models"""
    
    # Set the random seed for reproducible results
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    
    # Set the device
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    # Test AttentionModel if checkpoint path is provided
    if hasattr(opts, 'am_checkpoint_path') and opts.am_checkpoint_path:
        print("=" * 60)
        print("Testing AttentionModel...")
        print("=" * 60)
        
        try:
            am_model = load_attention_model(opts.am_checkpoint_path, opts)
            
            trainer = RL4COTrainer(
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                logger=False,
            )
            
            am_out = trainer.test(am_model)
            am_reward = am_out[0]["test/reward"] * opts.graph_size * -1
            results['AttentionModel'] = am_reward
            
            print(f"AttentionModel Test Reward: {am_reward:.4f}")
            
        except Exception as e:
            print(f"Error testing AttentionModel: {e}")
            results['AttentionModel'] = None
    
    # Test NeuOpt if checkpoint path is provided
    if hasattr(opts, 'neuopt_checkpoint_path') and opts.neuopt_checkpoint_path:
        print("=" * 60)
        print("Testing NeuOpt...")
        print("=" * 60)
        
        try:
            neuopt_model = load_neuopt_model(opts.neuopt_checkpoint_path, opts)
            
            trainer = RL4COTrainer(
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                logger=False,
            )
            
            neuopt_out = trainer.test(neuopt_model)
            neuopt_cost = neuopt_out[0]["test/cost_bsf"]
            results['NeuOpt'] = neuopt_cost
            
            print(f"NeuOpt Test Cost BSF: {neuopt_cost:.4f}")
            
        except Exception as e:
            print(f"Error testing NeuOpt: {e}")
            results['NeuOpt'] = None
    
    # Print comparison results
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        if result is not None:
            print(f"{model_name}: {result:.4f}")
        else:
            print(f"{model_name}: Failed to test")
    
    if len(results) == 2 and all(v is not None for v in results.values()):
        am_result = results.get('AttentionModel')
        neuopt_result = results.get('NeuOpt')
        
        if am_result is not None and neuopt_result is not None:
            print(f"\nDifference (AttentionModel - NeuOpt): {am_result - neuopt_result:.4f}")


    ks = [1, 2, 3, 4, 5][::-1]
    print(f"AttentionModel constructive length:\t {am_result:.2f},\t time: <1s \t(GPU in parallel)")
    print(f"NeuOpt constructive length:\t {neuopt_result:.2f},\t time: <1s \t(GPU in parallel)")
    print('-' * 90)

    for k in ks:
        greedy_baseline, t = test_greedy(opts, k)
        gap_am = (greedy_baseline - am_result) / am_result * 100
        gap_no = (greedy_baseline - neuopt_result) / neuopt_result * 100
        print(f"{k}-mers-greedy length:\t {greedy_baseline:.2f},\tgap_am: {gap_am:.2f}%,\tgap_no: {gap_no:.2f}%,\t time: {t} \t(CPU in series)")

    return results


def generate_test_file(opts):
    env = SSPkoptEnv(generator_params=dict(num_loc=opts.graph_size, fixed_len=opts.string_length, init_sol_type=opts.init_val_met), k_max=opts.k)
    batch_size = opts.batch_size
    td = env.reset(batch_size=[batch_size])
    
    # Extract codes from the tensor dict
    codes_data = td["codes"].cpu().numpy()
    cost_matrix = td["cost_matrix"].cpu().numpy()

    # Create data directory if it doesn't exist
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to npz file
    np.savez(os.path.join(data_dir, "data_ssp.npz"), codes=codes_data, cost_matrix=cost_matrix)
    print(f"Data saved to {os.path.join(data_dir, 'data_ssp.npz')}")
    print(f"Codes shape: {codes_data.shape}")

def run_test(opts):
    """Main function to run the test"""

    if opts.test_file is None:
        generate_test_file(opts)
        opts.test_file = "data_ssp.npz"
    
    # Pretty print the run args
    print("Test Configuration:")
    pprint.pprint(vars(opts))
    
    # Run the comparison test
    results = test_models(opts)
    
    # Save results if needed
    if not opts.no_saving:
        results_path = os.path.join('.', 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        


if __name__ == "__main__":
    opts = get_options()
    
    # Add checkpoint paths for testing (modify these paths as needed)
    if not hasattr(opts, 'am_checkpoint_path'):
        opts.am_checkpoint_path = None  # Set path to AttentionModel checkpoint
    if not hasattr(opts, 'neuopt_checkpoint_path'):
        opts.neuopt_checkpoint_path = None  # Set path to NeuOpt checkpoint
    
    # You can also set the checkpoint paths via command line or modify here:
    # opts.am_checkpoint_path = "path/to/attention_model.ckpt"
    # opts.neuopt_checkpoint_path = "path/to/neuopt_model.ckpt"
    
    if not opts.am_checkpoint_path and not opts.neuopt_checkpoint_path:
        print("Please provide checkpoint paths for testing:")
        print("Set opts.am_checkpoint_path for AttentionModel")
        print("Set opts.neuopt_checkpoint_path for NeuOpt")
        print("Example usage:")
        print("python test.py --am_checkpoint_path path/to/am.ckpt --neuopt_checkpoint_path path/to/neuopt.ckpt")
    else:
        run_test(opts)