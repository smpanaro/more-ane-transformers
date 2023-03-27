from vanilla_gpt2 import GPT as Vanilla
from ane_gpt2 import GPT as ANE
import torch
import numpy as np
from psnr import compute_psnr

"""
What is this file?
It's a step-by-step, layer-by-layer process of debugging why the Apple Neural Engine (ANE) optimized
model does not match the reference (aka vanilla) gpt2 model.

The PSNR (peak signal-to-noise ratio) is taken from the ane transformers repo and is a way
to measure how well the two models match. It seems like 150-200 is good (since these should be mathematically equivalent
-- e.g. in the Apple repo 60 is the minimum bar they set for the lossy conversion to CoreML).
Despite some horrific bugs I never got lower than 20 or so. Lower than 60 and something is likely wrong.

Run this file, look at the PSNR logs and if any are <<150, start debugging that layer.
"""

ane = ANE.from_pretrained("gpt2")
van = Vanilla.from_pretrained("gpt2")
ane.eval()
van.eval()

seqlen = 20
idx = torch.randint(30000, (1, seqlen,))
ane_inputs = ane.build_inputs(idx)
qk_mask, k_mask = [ane_inputs[k] for k in ["qk_mask", "k_mask"]]
k_mask = None

print("---\n")

def print_stats(name: str, a, v, should_be_equal=False, end="\n"):
    if torch.equal(a,v) and not should_be_equal:
        print("BUG: most likely passed the wrong things")
    print(name, a.shape, v.shape, "psnr", compute_psnr(a, v), end)

def bc1s_to_bsc(x):
    assert len(x.shape) == 4
    assert x.shape[0] == 1 # batch is 1
    assert x.shape[2] == 1 # third dim is always 1
    return x.permute(0, 3, 1, 2).squeeze(-1)

with torch.no_grad():
    ane_embd = ane.transformer.wte(idx)
    van_embd = van.transformer.wte(idx)
    print("wte", ane_embd.shape, van_embd.shape, torch.equal(ane_embd, van_embd))

    ane_preb = ane.prepare_for_block(idx)
    van_preb = van.prepare_for_block(idx)
    print_stats("pre block", bc1s_to_bsc(ane_preb), van_preb)

    ane_b1 = ane.transformer.h[0](ane.prepare_for_block(idx), qk_mask=qk_mask, k_mask=k_mask)
    van_b1 = van.transformer.h[0](van.prepare_for_block(idx))
    print_stats("layer 1", bc1s_to_bsc(ane_b1), van_b1)

    ane_b1_ln_1 = ane.transformer.h[0].ln_1(ane.prepare_for_block(idx))
    van_b1_ln_1 = van.transformer.h[0].ln_1(van.prepare_for_block(idx))
    print_stats("layer 1 ln1", bc1s_to_bsc(ane_b1_ln_1), van_b1_ln_1,)

    ane_b1_attn = ane.transformer.h[0].attn(ane_b1_ln_1, qk_mask=qk_mask, k_mask=k_mask)
    van_b1_attn = van.transformer.h[0].attn(van_b1_ln_1)
    print_stats("layer 1 attn", bc1s_to_bsc(ane_b1_attn), van_b1_attn)

    ane_b1_attn_qproj = ane.transformer.h[0].attn.q_proj(ane_b1_ln_1)
    van_b1_attn_qproj = van.transformer.h[0].attn.q_proj(van_b1_ln_1)
    print_stats("layer 1 qproj", bc1s_to_bsc(ane_b1_attn_qproj), van_b1_attn_qproj)

    ane_b1_attn_kproj = ane.transformer.h[0].attn.k_proj(ane_b1_ln_1)
    van_b1_attn_kproj = van.transformer.h[0].attn.k_proj(van_b1_ln_1)
    print_stats("layer 1 kproj", bc1s_to_bsc(ane_b1_attn_kproj), van_b1_attn_kproj)

    ane_b1_attn_vproj = ane.transformer.h[0].attn.v_proj(ane_b1_ln_1)
    van_b1_attn_vproj = van.transformer.h[0].attn.v_proj(van_b1_ln_1)
    print_stats("layer 1 vproj", bc1s_to_bsc(ane_b1_attn_vproj), van_b1_attn_vproj)

    ane_b1_attn_atnn = ane.transformer.h[0].attn._attention_fn(ane_b1_attn_qproj, ane_b1_attn_kproj, ane_b1_attn_vproj, qk_mask, k_mask, False)[0].contiguous().view(1, 768, 1, seqlen)
    van_b1_attn_atnn = van.transformer.h[0].attn._attention_fn(van_b1_ln_1, van_b1_attn_qproj, van_b1_attn_kproj, van_b1_attn_vproj)
    print_stats("layer 1 inner attn", bc1s_to_bsc(ane_b1_attn_atnn), van_b1_attn_atnn)

    # Not sure I fully trust this due to the change to add cat in ane + the permute.
    ane_b1_attn_qxk = ane.transformer.h[0].attn._qk(ane_b1_attn_qproj, ane_b1_attn_kproj, ane_b1_attn_vproj)
    van_b1_attn_qxk = van.transformer.h[0].attn._qk(van_b1_ln_1, van_b1_attn_qproj, van_b1_attn_kproj, van_b1_attn_vproj)
    assert len(ane_b1_attn_qxk) == len(van_b1_attn_qxk)
    print_stats("layer 1 q@k", ane_b1_attn_qxk.permute(0, 2, 3, 1), van_b1_attn_qxk)

    ane_b1_attn_qxk_sm = ane.transformer.h[0].attn._qk_softmax(ane_b1_attn_qproj, ane_b1_attn_kproj, ane_b1_attn_vproj, qk_mask=qk_mask, k_mask=k_mask)
    van_b1_attn_qxk_sm = van.transformer.h[0].attn._qk_softmax(van_b1_ln_1, van_b1_attn_qproj, van_b1_attn_kproj, van_b1_attn_vproj)
    assert len(ane_b1_attn_qxk_sm) == len(van_b1_attn_qxk_sm)
    print_stats("layer 1 q@k softmax", ane_b1_attn_qxk_sm.permute(0, 2, 3, 1), van_b1_attn_qxk_sm)

    # Bubble back up to middle of block.
    ane_b1_x_post_attn = ane_b1_ln_1 + ane_b1_attn
    van_b1_x_post_attn = van_b1_ln_1 + van_b1_attn
    print_stats("layer 1 x + attn", ane_b1_attn_qxk_sm.permute(0, 2, 3, 1), van_b1_attn_qxk_sm)

    ane_b1_ln2 = ane.transformer.h[0].ln_2(ane_b1_x_post_attn)
    van_b1_ln2 = van.transformer.h[0].ln_2(van_b1_x_post_attn)
    print_stats("layer 1 ln2", bc1s_to_bsc(ane_b1_ln2), van_b1_ln2)

    ane_b1_mlp = ane.transformer.h[0].mlp(ane_b1_ln2)
    van_b1_mlp = van.transformer.h[0].mlp(van_b1_ln2)
    print_stats("layer 1 mlp", bc1s_to_bsc(ane_b1_mlp), van_b1_mlp)

    ane_all_blocks = ane.prepare_for_block(idx)
    van_all_blocks = van.prepare_for_block(idx)
    for nh, (ab, vb) in enumerate(zip(ane.transformer.h, van.transformer.h)):
        ane_all_blocks = ab(ane_all_blocks, qk_mask=qk_mask, k_mask=k_mask)
        van_all_blocks = vb(van_all_blocks)
        print_stats(f"layer {nh}", bc1s_to_bsc(ane_all_blocks), van_all_blocks, end="")

    ane_lm = ane.lm_head(ane_all_blocks.permute(3, 1, 0, 2)).squeeze().unsqueeze(0)
    van_lm = van.lm_head(van_all_blocks)
    print_stats("lm", ane_lm, van_lm)

# E2E - with k_mask to make sure that works.

ane_inputs = ane.build_inputs(idx, pad_to_length=40, pad_token_id=350)
ane_idx = ane_inputs["input_ids"]
qk_mask, k_mask, output_mask = [ane_inputs[k] for k in ["qk_mask", "k_mask", "output_mask"]]

with torch.no_grad():
    ane_out = ane(ane_idx, qk_mask=qk_mask, output_mask=output_mask, k_mask=None)[:, [-1], :]
    van_out = van(idx)[0]

assert ane_out.shape == van_out.shape, f"{ane_out.shape} != {van_out.shape}"

print("shapes:", van_out.shape, ane_out.shape)
print("softmax argmax:", torch.argmax(van_out.softmax(2)), torch.argmax(ane_out.softmax(2)))
print("e2e psnr", compute_psnr(ane_out.softmax(2).numpy(), van_out.softmax(2).numpy()))
# sim = torch.nn.functional.cosine_similarity(van_out[0], ane_out[0])
# print("similarity", sim, torch.mean(sim)) # Turns out this is mostly useless, it's quite easy to get upper 0.9s.