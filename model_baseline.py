import jax
import jax.numpy as jnp
import equinox as eqx

from model_utils import Encoder, SpatialBroadcastDecoder

# iterative slot attention
class SlotAttention(eqx.Module):
    # hyperparameters
    num_slots: int
    slot_dim: int
    num_iter: int
    scale: float

    # learnable slot initialization
    slots_mu: jax.Array # [1, 1, slot_dim]
    slots_log_sigma: jax.Array # [1, 1, slot_dim]

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

    def __init__(self, num_slots: int, slot_dim: int, enc_dim: int, num_iter: int = 3, mlp_hidden: int = 128, *, key: jax.Array):
        keys = jax.random.split(key, 8)

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iter = num_iter
        self.scale = slot_dim ** -0.5

        self.slots_mu = jax.random.normal(keys[0], (1, 1, slot_dim))
        self.slots_log_sigma = jnp.zeros((1, 1, slot_dim))

        self.norm_input = eqx.nn.LayerNorm(enc_dim)
        self.fc_input = eqx.nn.Linear(enc_dim, slot_dim, key=keys[1])

        self.to_q = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[2])
        self.to_k = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[3])
        self.to_v = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=keys[4])

        self.gru = eqx.nn.GRUCell(slot_dim, slot_dim, key=keys[5])

        self.norm_slots = eqx.nn.LayerNorm(slot_dim)
        self.norm_pre_ff = eqx.nn.LayerNorm(slot_dim)
        self.mlp_0 = eqx.nn.Linear(slot_dim, mlp_hidden, key=keys[6])
        self.mlp_1 = eqx.nn.Linear(mlp_hidden, slot_dim, key=keys[7])

    def initialize_slots(self, batch_size: int, key: jax.Array) -> jax.Array: # sample initial slots from a learned gaussian
        mu = jnp.broadcast_to(self.slots_mu, (batch_size, self.num_slots, self.slot_dim)) # [B, N_slots, slot_dim]
        sigma = jnp.broadcast_to(jnp.exp(self.slots_log_sigma), (batch_size, self.num_slots, self.slot_dim)) # [B, N_slots, slot_dim]
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise # [B, N_slots, slot_dim]

    def __call__(self, enc_feat: jax.Array, key: jax.Array) -> jax.Array:
        """
        enc_feat: [B, N_feat, D_encoder]
        key: PRNG key for slot initialization

        returns: slots [B, N_slots, slot_dim]
        """
        B = enc_feat.shape[0]

        # project input features once, precompute K and V
        inputs_norm = jax.vmap(jax.vmap(self.norm_input))(enc_feat)
        inputs = jax.vmap(jax.vmap(self.fc_input))(inputs_norm)
        k = jax.vmap(jax.vmap(self.to_k))(inputs) # [B, N_feat, slot_dim]
        v = jax.vmap(jax.vmap(self.to_v))(inputs) # [B, N_feat, slot_dim]

        # initialize slots
        slots = self.initialize_slots(B, key) # [B, N_slots, slot_dim]

        # iterative refinement
        # a plain Python for loop is fine here since num_iter is small and fixed (jax.lax.fori_loop would be better for JIT efficiency)
        for _ in range(self.num_iter):
            slots_prev = slots

            # normalize and compute Q
            slots_norm = jax.vmap(jax.vmap(self.norm_slots))(slots)
            q = jax.vmap(jax.vmap(self.to_q))(slots_norm) # [B, N_slots, slot_dim]

            # attention with the softmax applied over slots
            att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
            att = jax.nn.softmax(att_logits, axis=1) # [B, N_slots, N_feat]
            att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)

            updates = jnp.einsum('bnm,bmd->bnd', att, v) # [B, N_slots, slot_dim]

            # GRU update, eqx.nn.GRUCell takes (input, hidden) for a single vector, vmap over batch and slot dimensions
            def gru_step(update, hidden):
                return self.gru(update, hidden)

            slots = jax.vmap(jax.vmap(gru_step))(updates, slots_prev)

            # residual MLP
            slots_normed = jax.vmap(jax.vmap(self.norm_pre_ff))(slots)
            h = jax.vmap(jax.vmap(self.mlp_0))(slots_normed)
            h = jax.nn.relu(h)
            h = jax.vmap(jax.vmap(self.mlp_1))(h)
            slots = slots + h

        return slots # [B, N_slots, slot_dim]

# slot attention model (baseline)
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

    enc: Encoder
    slot_attention: SlotAttention
    dec: SpatialBroadcastDecoder

    def __init__(self, resolution: tuple = (64, 64), num_slots: int = 7, slot_dim: int = 64, enc_hidden_dim: int = 64, num_iter: int = 3, *, key: jax.Array):
        k_enc, k_sa, k_dec = jax.random.split(key, 3)

        self.resolution = resolution
        self.num_slots = num_slots

        self.enc = Encoder(enc_hidden_dim, key=k_enc)

        self.slot_attention = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, enc_dim=enc_hidden_dim, num_iter=num_iter, key=k_sa)

        self.dec = SpatialBroadcastDecoder(slot_dim, resolution, dec_hidden_dim=enc_hidden_dim, key=k_dec)

    def __call__(self, image: jax.Array, *, key: jax.Array) -> tuple:
        """
        image: [B, 3, H, W]
        key: PRNG key for slot initialization

        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
            slots: [B, N_slots, slot_dim]
        """
        enc_feat = self.enc(image)
        slots = self.slot_attention(enc_feat, key) # [B, N_slots, slot_dim]
        recon, masks = self.dec(slots) # [B, 3, H, W], [B, N_slots, H, W]
        return recon, masks, slots