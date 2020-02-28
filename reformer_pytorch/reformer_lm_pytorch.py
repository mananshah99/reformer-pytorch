import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial
from itertools import chain
from reversible import ReversibleBlock, ReversibleSequence
from reformer_pytorch import *

from recorder import Recorder

# DONE: fix dimensions so there aren't hacky 2s everywhere (this is due to recurrence via Transformer XL)
# TODO: figure out how relative positional encoding would work with this kind of setup
# TODO: run baseline evaluations on these tasks (using full QK attention and LSH attention, turning recurrence on and off)
# TODO: write up report for the milestone with graphs and stuff
class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, recurrence = False, heads = 8, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, use_full_attn = False, full_attn_thres = 0, num_mem_kv = 0, emb_dim = None, return_embeddings = False, fixed_position_emb = False):
        super().__init__()

        # 1. Get embeddings
        emb_dim = default(emb_dim, dim)
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = FixedPositionEmbedding(emb_dim) if fixed_position_emb else nn.Embedding(max_seq_len, emb_dim)

        # 2. Projection from embedding dimension to Reformer input dimension
        self.to_model_dim = identity if emb_dim == dim else nn.Linear(emb_dim, dim)

        # 3. Reformer model
        self.reformer = Reformer(dim, 
                                  depth, 
                                  max_seq_len, 
                                  recurrence = recurrence, 
                                  heads = heads, 
                                  bucket_size = bucket_size, 
                                  n_hashes = n_hashes, 
                                  ff_chunks = ff_chunks, 
                                  attn_chunks = attn_chunks, 
                                  causal = causal, 
                                  weight_tie = weight_tie, 
                                  lsh_dropout = lsh_dropout, 
                                  layer_dropout = layer_dropout, 
                                  random_rotations_per_head = random_rotations_per_head, 
                                  twin_attention = twin_attention, 
                                  use_scale_norm = use_scale_norm, 
                                  use_full_attn = use_full_attn, 
                                  full_attn_thres = full_attn_thres, 
                                  num_mem_kv = num_mem_kv)

        self.reformer.turn_on()
        
        # 4. Function to return embeddings / probabilities
        self.to_logits = identity if return_embeddings else nn.Linear(dim * 2 if recurrence else dim, num_tokens)

    def forward(self, x, **kwargs):

        # 1. Add position embeddings
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x)
        x = x + self.pos_emb(t).type(x.type())

        # 2. Project to model dimension and run Reformer
        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)

        #attn_weights_and_buckets = self.reformer.recordings[0]
        #print(len(attn_weights_and_buckets))
        #print(attn_weights_and_buckets[0].keys())
        #print([x.shape for x in attn_weights_and_buckets[0].values()])
        
        # 3. Return embeddings / probabilities
        return self.to_logits(x)