#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn

from enum import Enum

from .layer_norm import LayerNormANE

class AttentionImplementations(Enum):
    #ORIGINAL = "ORIGINAL"
    EINSUM = "EINSUM"
    SPLIT_EINSUM = "SPLIT_EINSUM"

ATTENTION_IMPLEMENTATION_IN_EFFECT = AttentionImplementations.SPLIT_EINSUM

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention optimized for efficient ANE deployment
    """

    def __init__(self,
                 embed_dim,
                 d_qk=None,
                 d_v=None,
                 d_out=None,
                 n_head=8,
                 dropout=0.1,
                 **kwargs):
        """
        Args:
            embed_dim:          Dimensionality of the input embeddings
            d_qk:               Dimensionality of the query and key embeddings. They must match in order to compute
                                dot product attention. If None, it is set to that of the input tensors, i.e. `embed_dim`
            d_v:                Dimensionality of the value embeddings. It may differ from that of the query and
                                key embeddings. If None, it is set to that of the input tensors.
            d_out:              Dimensionality of the output projection. If None, it is set to that of the input tensors
            n_head:             The number of different attention heads to compute in parallel, uses :math:`d_qk/n_head`
                                channel groups in parallel to learn different attention patterns in a single layer
            dropout:            The probability that each attention weight is zero-masked independent of other weights
        """
        super().__init__()

        self.d_qk = d_qk or embed_dim
        self.d_v = d_v or embed_dim
        self.d_out = d_out or embed_dim

        self.n_head = n_head
        if self.d_qk % self.n_head != 0 or self.d_v % self.n_head != 0:
            raise ValueError(
                f"Either query-key dimensions ({self.d_qk}) or the value embeddings "
                f"dimensions ({self.d_v}) is not divisible by n_head ({self.n_head})"
            )
        self.q_normalize_fact = float(self.d_qk // self.n_head)**-0.5

        self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def _qk(self, q, k, v):
        # Principle 2: Chunking Large Intermediate Tensors  (machinelearning.apple.com/research/apple-neural-engine)
        # Split q, k and v to compute a list of single-head attention functions
        mh_q = q.split(
            self.d_qk // self.n_head,
            dim=1)  # n_head * (batch_size, d_qk/n_head, 1, tgt_seq_len)
        # Principle 3: Minimizing Memory Copies
        # Avoid as many transposes and reshapes as possible
        mh_k = k.transpose(1, 3).split(
            self.d_qk // self.n_head,
            dim=3)  # n_head * (batch_size, src_seq_len, 1, d_qk/n_head)
        mh_v = v.split(
            self.d_v // self.n_head,
            dim=1)  # n_head * (batch_size, d_v/n_head, 1, src_seq_len)

        # `qk = q @ k`
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * self.q_normalize_fact
            for qi, ki in zip(mh_q, mh_k)
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        attn_weights = torch.cat(attn_weights, dim=2)
        return attn_weights

    def _qk_softmax(self, q, k, v, qk_mask=None, k_mask=None):
        # Principle 2: Chunking Large Intermediate Tensors  (machinelearning.apple.com/research/apple-neural-engine)
        # Split q, k and v to compute a list of single-head attention functions

        # Do the scalar multiply before the matmul so it can be folded into the conv.
        q = q * self.q_normalize_fact

        mh_q = q.split(
            self.d_qk // self.n_head,
            dim=1)  # n_head * (batch_size, d_qk/n_head, 1, tgt_seq_len)
        # Principle 3: Minimizing Memory Copies
        # Avoid as many transposes and reshapes as possible
        mh_k = k.transpose(1, 3).split(
            self.d_qk // self.n_head,
            dim=3)  # n_head * (batch_size, src_seq_len, 1, d_qk/n_head)
        mh_v = v.split(
            self.d_v // self.n_head,
            dim=1)  # n_head * (batch_size, d_v/n_head, 1, src_seq_len)

        # `qk = q @ k`
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        # Apply attention masking
        if qk_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
                # attn_weights[head_idx] = attn_weights[head_idx].masked_fill(torch.tril(torch.ones(5, 5)).t().view(1,5,1,5) == 0, float("-inf"))#-1e4)
        if k_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + k_mask

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights
                        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)
        mh_w = [self.dropout(aw) for aw in attn_weights
                ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        # (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(mh_w, mh_v)
        ]  # n_head * (batch_size, d_v/n_head, 1, tgt_seq_len)
        attn = torch.cat(attn, dim=2)  # (batch_size, d_v, 1, tgt_seq_len)
        return attn

    def _attention_fn(self, q, k, v, qk_mask, k_mask, return_weights):
        """Core routine for computing multi-head attention

        Args:
            q:              Projected query embeddings of shape (batch_size, d_qk, 1, tgt_seq_len)
            k:              Projected key embeddings of shape (batch_size, d_qk, 1, src_seq_len)
            v:              Projected value embeddings of shape (batch_size, d_v, 1, src_seq_len)
            qk_mask:        Float tensor of shape (batch_size, src_seq_len, 1, tgt_seq_len).
                            Indices with the a high negative value, e.g. -1e4, are excluded from attention
            k_mask:         Float tensor of shape (batch_size, src_seq_len, 1, 1).
                            Indices with the a high negative value, e.g. -1e4, are excluded from attention

        Returns:
            attn:           Attention embeddings of shape (batch_size, d_v, 1, tgt_seq_len)
            attn_weights:   If `return_weights` is True, returns the softmax attention weights used to compute the attention matrix
        """
        if ATTENTION_IMPLEMENTATION_IN_EFFECT == AttentionImplementations.EINSUM:
            # Stolen/inspired from ml-stable-diffusion repo. No clue if it works,
            # had a bug that turned it off by mistake so never tried.
            bs = q.size(0)
            dim_head = self.d_qk // self.n_head
            mh_q = q.view(bs, self.n_head, dim_head, -1)
            mh_k = k.view(bs, self.n_head, dim_head, -1)
            mh_v = v.view(bs, self.n_head, dim_head, -1)

            # print("qkv", q.shape, k.shape, v.shape)
            # print("mhqkv", mh_q.shape, mh_k.shape, mh_v.shape)

            attn_weights = torch.einsum("bhcq,bhck->bhqk", [mh_q, mh_k]) # 1,64,12,20 @ 1,64,20,12 = 1,64,20,20
            attn_weights.mul_(self.q_normalize_fact)

            # print("attn,qk", attn_weights.shape, qk_mask.shape, k_mask.shape if k_mask is not None else None)

            if qk_mask is not None:
                qk_mask = qk_mask.squeeze(2).unsqueeze(0)
                attn_weights = attn_weights + qk_mask
            if k_mask is not None:
                k_mask = k_mask.squeeze(2).unsqueeze(0)
                attn_weights = attn_weights + k_mask

            attn_weights = attn_weights.softmax(dim=3)

            attn = torch.einsum("bhqk,bhck->bhcq", [attn_weights, mh_v])
            attn = attn.contiguous().view(bs, self.d_out, 1, -1)
        elif ATTENTION_IMPLEMENTATION_IN_EFFECT == AttentionImplementations.SPLIT_EINSUM:
            # Do the scalar multiply before the matmul so it can be folded into the conv.
            q = q * self.q_normalize_fact

            # Principle 2: Chunking Large Intermediate Tensors  (machinelearning.apple.com/research/apple-neural-engine)
            # Split q, k and v to compute a list of single-head attention functions
            mh_q = q.split(
                self.d_qk // self.n_head,
                dim=1)  # n_head * (batch_size, d_qk/n_head, 1, tgt_seq_len)
            # Principle 3: Minimizing Memory Copies
            # Avoid as many transposes and reshapes as possible
            mh_k = k.transpose(1, 3).split(
                self.d_qk // self.n_head,
                dim=3)  # n_head * (batch_size, src_seq_len, 1, d_qk/n_head)
            mh_v = v.split(
                self.d_v // self.n_head,
                dim=1)  # n_head * (batch_size, d_v/n_head, 1, src_seq_len)

            # `qk = q @ k`
            attn_weights = [
                torch.einsum('bchq,bkhc->bkhq', [qi, ki])
                for qi, ki in zip(mh_q, mh_k)
            ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

            # Apply attention masking
            if qk_mask is not None:
                for head_idx in range(self.n_head):
                    attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
            if k_mask is not None:
                for head_idx in range(self.n_head):
                    attn_weights[head_idx] = attn_weights[head_idx] + k_mask

            # `w = F.softmax(qk.float(), dim=-1)`
            attn_weights = [aw.softmax(dim=1) for aw in attn_weights
                            ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)
            mh_w = [self.dropout(aw) for aw in attn_weights
                    ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

            # (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            attn = [
                torch.einsum('bkhq,bchk->bchq', wi, vi)
                for wi, vi in zip(mh_w, mh_v)
            ]  # n_head * (batch_size, d_v/n_head, 1, tgt_seq_len)
            attn = torch.cat(attn, dim=1)  # (batch_size, d_v, 1, tgt_seq_len)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def _forward_impl(
        self,
        q,
        k,
        v,
        qpos=None,
        kpos=None,
        vpos=None,
        qk_mask=None,
        k_mask=None,
        return_weights=True,
    ):
        """
        Args:
            q:                  Query embeddings of shape (batch_size, embed_dim, 1, tgt_seq_len)
            k:                  Key embeddings of shape (batch_size, embed_dim, 1, src_seq_len)
            v:                  Value embeddings of shape (batch_size, embed_dim, 1, src_seq_len)
            qpos:               Positional encodings for the query embeddings with same shape as `q`
            kpos:               Positional encodings for the key embeddings with same shape as `k`
            vpos:               Positional encodings for the key embeddings with same shape as `v`
            qk_mask:            Float tensor with shape (batch_size, src_seq_len, 1, tgt_seq_len). Example use case: for causal masking
                                in generative language models (e.g. GPT), fill the upper triangular part with a high negative value (e.g. -1e4).
                                Indices with the a high negative value, e.g. -1e4, are excluded from attention
            k_mask:             Float tensor with shape (batch_size, src_seq_len, 1, 1). Example use case: when excluding embeddings that
                                correspond to zero-padded pixels in an image or unused tokens in a text token sequence from attention.
                                Indices with the a high negative value, e.g. -1e4, are excluded from attention
            return_weights:     If True, returns the intermediate attention weights


        Note: If any of q,k,v has shape (batch_size, embed_dim, height, width) that represent a 2-d feature map, this will
        be flattened to (batch_size, embed_dim, 1, height * width)

        Note: `attn_weights` are never passed downstream even when return_weights=True because all the attn_weights
        are harvested from the outermost module (e.g. ane_transformers.model#Transformer) by means of forward hooks
        """
        # Parse tensor shapes for source and target sequences
        assert len(q.size()) == 4 and len(k.size()) == 4 and len(v.size()) == 4, f"q:{q.size()} k:{k.size()} v:{v.size()}"
        b, ct, ht, wt = q.size()
        b, cs, hs, ws = k.size()

        tgt_seq_len = ht * wt
        src_seq_len = hs * ws

        # Add positional encodings if any
        if qpos is not None:
            q = q + qpos
        if kpos is not None:
            k = k + kpos
        if vpos is not None:
            v = v + kpos

        # Project q,k,v
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Validate qk_mask (`attn_mask` in `torch.nn.MultiheadAttention`)
        expected_qk_mask_shape = [b, src_seq_len, 1, tgt_seq_len]
        if qk_mask is not None:
            if qk_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`qk_mask` must be of type torch.float32, received {qk_mask.dtype}"
                )
            if list(qk_mask.size()) != expected_qk_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `qk_mask` (Expected {expected_qk_mask_shape}, got {list(qk_mask.size())}"
                )

        # Validate k_mask (`key_padding_mask` in `torch.nn.MultiheadAttention`)
        expected_k_mask_shape = [b, src_seq_len, 1, 1]
        if k_mask is not None:
            if k_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`k_mask` must be of type torch.float32, received {k_mask.dtype}"
                )
            if list(k_mask.size()) != expected_k_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `k_mask` (Expected {expected_k_mask_shape}, got {list(k_mask.size())}"
                )

        # Call the attention function
        attn, attn_weights = self._attention_fn(q, k, v, qk_mask, k_mask,
                                                return_weights)

        # Revert to original dimension permutation
        attn = attn.contiguous().view(b, self.d_v, ht, wt)

        attn = self.out_proj(attn)

        # if return_weights:
        #     return attn, attn_weights
        return attn #, None

    def forward(self, q, k, v, **kwargs):
        return self._forward_impl(q, k, v, **kwargs)


class ResidualMultiHeadAttention(MultiHeadAttention):

    def __init__(self, embed_dim, dropout=0.1, drop_fn=nn.Dropout, **kwargs):
        super().__init__(embed_dim, dropout=dropout, **kwargs)
        self.drop_fn = drop_fn(dropout) if dropout > 0. else nn.Identity()
        self.norm = LayerNormANE(embed_dim)

    def forward(self, q, k, v, **kwargs):
        attn, attn_weights = self._forward_impl(q, k, v, **kwargs)
        return self.norm(self.drop_fn(attn) + q), attn_weights


class SelfAttention(MultiHeadAttention):

    def forward(self, qkv, **kwargs):
        return super()._forward_impl(qkv, qkv, qkv, **kwargs)


class ResidualSelfAttention(ResidualMultiHeadAttention):

    def forward(self, qkv, **kwargs):
        attn, attn_weights = self._forward_impl(qkv, qkv, qkv, **kwargs)
        return self.norm(self.drop_fn(attn) + qkv), attn_weights


class PreNormResidualSelfAttention(ResidualSelfAttention):

    def forward(self, qkv, **kwargs):
        norm_qkv = self.norm(qkv)
        attn, attn_weights = self._forward_impl(norm_qkv, norm_qkv, norm_qkv,
                                                **kwargs)
        result = self.drop_fn(attn) + qkv
        return result, attn_weights