"""
Baseline Slot Attention Model — JAX / Equinox implementation.
(Locatello et al., 2020)

Faithful reimplementation of the original Slot Attention paper at 64x64.
Same encoder, decoder, and positional embeddings as the SlotODE model —
only the slot update mechanism differs (iterative GRU refinement vs ODE).

Key difference from SlotODE:
  - Slots are refined in discrete iterations (default 3)
  - Each iteration: compute Q from current slots, attend to K/V, update
    via GRU, then apply a residual MLP
  - K, V are computed once from input features (not per iteration)

See model.py header for JAX/Equinox/Diffrax primer.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

# reuse shared components from the ODE model
from model import SoftPositionalEmbedding, SpatialBroadcastDecoder


# ---------------------------------------------------------------------------
# Iterative Slot Attention (original paper)
# ---------------------------------------------------------------------------

class SlotAttention(eqx.Module):
    """Original iterative Slot Attention module.

    Each iteration:
      1. LayerNorm slots, compute Q
      2. Attend to precomputed K, V (softmax over slot dim = competition)
      3. GRU update: new_slot = GRU(attended, old_slot)
      4. Residual MLP: slot = slot + MLP(LayerNorm(slot))
    """
    num_slots: int
    slot_dim: int
    num_iter: int
    scale: float

    # learnable slot initialization
    slots_mu: jax.Array         # [1, 1, slot_dim]
    slots_log_sigma: jax.Array  # [1, 1, slot_dim]

    # input projection
    norm_input: eqx.nn.LayerNorm
    fc_input: eqx.nn.Linear

    # attention projections
    to_q: eqx.nn.Linear
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear

    # GRU for slot update
    gru: eqx.nn.GRUCell

    # residual MLP
    norm_slots: eqx.nn.LayerNorm
    norm_pre_ff: eqx.nn.LayerNorm
    mlp_0: eqx.nn.Linear
    mlp_1: eqx.nn.Linear

    def __init__(self, num_slots: int, slot_dim: int, enc_dim: int,
                 num_iter: int = 3, mlp_hidden: int = 128, *, key: jax.Array):
        keys = jax.random.split(key, 8)

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iter = num_iter
        self.scale = slot_dim ** -0.5

        # learnable init
        self.slots_mu = jax.random.normal(keys[0], (1, 1, slot_dim))
        self.slots_log_sigma = jnp.zeros((1, 1, slot_dim))

        # input projection
        self.norm_input = eqx.nn.LayerNorm(enc_dim)
        self.fc_input = eqx.nn.Linear(enc_dim, slot_dim, key=keys[1])

        # attention
        self.to_q = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[2])
        self.to_k = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[3])
        self.to_v = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[4])

        # GRU: input_size=slot_dim, hidden_size=slot_dim
        self.gru = eqx.nn.GRUCell(slot_dim, slot_dim, key=keys[5])

        # residual MLP
        self.norm_slots = eqx.nn.LayerNorm(slot_dim)
        self.norm_pre_ff = eqx.nn.LayerNorm(slot_dim)
        self.mlp_0 = eqx.nn.Linear(slot_dim, mlp_hidden, key=keys[6])
        self.mlp_1 = eqx.nn.Linear(mlp_hidden, slot_dim, key=keys[7])

    def initialize_slots(self, batch_size: int, key: jax.Array) -> jax.Array:
        """Sample initial slots from learned Gaussian. [B, N_slots, D]"""
        mu = jnp.broadcast_to(self.slots_mu, (batch_size, self.num_slots, self.slot_dim))
        sigma = jnp.broadcast_to(
            jnp.exp(self.slots_log_sigma),
            (batch_size, self.num_slots, self.slot_dim)
        )
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise

    def __call__(self, enc_feat: jax.Array, key: jax.Array) -> jax.Array:
        """
        enc_feat: [B, N_feat, D_encoder]
        key: PRNG key for slot initialization

        returns: slots [B, N_slots, D_slot]
        """
        B = enc_feat.shape[0]

        # project input features once
        inputs_norm = jax.vmap(jax.vmap(self.norm_input))(enc_feat)
        inputs = jax.vmap(jax.vmap(self.fc_input))(inputs_norm)  # [B, N_feat, D_slot]
        k = jax.vmap(jax.vmap(self.to_k))(inputs)               # [B, N_feat, D_slot]
        v = jax.vmap(jax.vmap(self.to_v))(inputs)               # [B, N_feat, D_slot]

        # initialize slots
        slots = self.initialize_slots(B, key)  # [B, N_slots, D_slot]

        # iterative refinement
        # in JAX we use a for loop (or jax.lax.fori_loop for JIT efficiency)
        # plain Python loop is fine here since num_iter is small and static
        for _ in range(self.num_iter):
            slots_prev = slots

            # normalize and compute Q
            slots_norm = jax.vmap(jax.vmap(self.norm_slots))(slots)
            q = jax.vmap(jax.vmap(self.to_q))(slots_norm)  # [B, N_slots, D_slot]

            # attention: softmax over slots (competition)
            att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
            att = jax.nn.softmax(att_logits, axis=1)   # [B, N_slots, N_feat]
            att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)

            updates = jnp.einsum('bnm,bmd->bnd', att, v)  # [B, N_slots, D_slot]

            # GRU update — eqx.nn.GRUCell takes (input, hidden) for a single vector
            # vmap over batch and slot dimensions
            def gru_step(update, hidden):
                return self.gru(update, hidden)

            slots = jax.vmap(jax.vmap(gru_step))(updates, slots_prev)

            # residual MLP
            slots_normed = jax.vmap(jax.vmap(self.norm_pre_ff))(slots)
            h = jax.vmap(jax.vmap(self.mlp_0))(slots_normed)
            h = jax.nn.relu(h)
            h = jax.vmap(jax.vmap(self.mlp_1))(h)
            slots = slots + h

        return slots


# ---------------------------------------------------------------------------
# Full Baseline Model
# ---------------------------------------------------------------------------

class SlotAttentionModel(eqx.Module):
    """Baseline Slot Attention autoencoder (Locatello et al., 2020).

    Same encoder and decoder as SlotODEModel — only the slot update
    mechanism differs (iterative GRU vs neural ODE).

    Usage:
        key = jax.random.key(0)
        model = SlotAttentionModel(key=key)

        key, subkey = jax.random.split(key)
        recon, masks, slots = model(images, key=subkey)
    """
    resolution: tuple
    num_slots: int

    # CNN encoder (identical to SlotODEModel)
    enc_conv0: eqx.nn.Conv2d
    enc_conv1: eqx.nn.Conv2d
    enc_conv2: eqx.nn.Conv2d
    enc_conv3: eqx.nn.Conv2d

    enc_pos: SoftPositionalEmbedding

    # encoder MLP (residual)
    enc_norm: eqx.nn.LayerNorm
    enc_fc0: eqx.nn.Linear
    enc_fc1: eqx.nn.Linear

    # slot attention (iterative)
    slot_attention: SlotAttention

    # decoder (identical to SlotODEModel)
    dec: SpatialBroadcastDecoder

    def __init__(self, resolution: tuple = (64, 64), num_slots: int = 7,
                 slot_dim: int = 64, enc_hidden_dim: int = 64,
                 num_iter: int = 3, *, key: jax.Array):
        (k_c0, k_c1, k_c2, k_c3, k_pos, k_f0, k_f1,
         k_sa, k_dec) = jax.random.split(key, 9)

        self.resolution = resolution
        self.num_slots = num_slots

        # CNN encoder
        self.enc_conv0 = eqx.nn.Conv2d(3, enc_hidden_dim, 5, padding=2, key=k_c0)
        self.enc_conv1 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c1)
        self.enc_conv2 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c2)
        self.enc_conv3 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c3)

        self.enc_pos = SoftPositionalEmbedding(enc_hidden_dim, key=k_pos)

        # encoder MLP
        self.enc_norm = eqx.nn.LayerNorm(enc_hidden_dim)
        self.enc_fc0 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f0)
        self.enc_fc1 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f1)

        # slot attention (iterative, original paper)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, slot_dim=slot_dim, enc_dim=enc_hidden_dim,
            num_iter=num_iter, key=k_sa
        )

        # decoder
        self.dec = SpatialBroadcastDecoder(
            slot_dim, resolution, dec_hidden_dim=enc_hidden_dim, key=k_dec
        )

    def encode(self, image: jax.Array) -> jax.Array:
        """image: [B, 3, H, W] -> [B, H*W, enc_hidden_dim]"""
        def encode_single(img):
            x = jax.nn.relu(self.enc_conv0(img))
            x = jax.nn.relu(self.enc_conv1(x))
            x = jax.nn.relu(self.enc_conv2(x))
            x = jax.nn.relu(self.enc_conv3(x))
            return x

        x = jax.vmap(encode_single)(image)
        x = self.enc_pos(x)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(0, 2, 1)

        h = jax.vmap(jax.vmap(self.enc_norm))(x)
        h = jax.vmap(jax.vmap(self.enc_fc0))(h)
        h = jax.nn.relu(h)
        h = jax.vmap(jax.vmap(self.enc_fc1))(h)
        enc_feat = x + h

        return enc_feat

    def __call__(self, image: jax.Array, *, key: jax.Array) -> tuple:
        """
        image: [B, 3, H, W]
        key: PRNG key for slot initialization

        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
            slots: [B, N_slots, D_slot]
        """
        enc_feat = self.encode(image)
        slots = self.slot_attention(enc_feat, key)
        recon, masks = self.dec(slots)
        return recon, masks, slots
