import math
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from model_utils import Encoder, SpatialBroadcastDecoder

"""
standard slot attention: slots ← slots + f(slots, features)

slot attention as an autonomous neural ODE, no dependence on time: d(slots)/dt = f(slots)

1. encoder: [B, 3, H, W] -> [B, N_feat, D_enc] <--- original to locatellos paper

2. slot init: slots_0, [B, N_slots, D_slot], learnable gaussian

3. scene representation is fixed over integration time - precompute K and v:
    k = to_k(feat)
    v = to_v(feat)

4. Slot attention:
    q = slots_norm @ Wq
    att = softmax(q @ k^T)
    f_attn = att @ v

5. Attention gate - how much of the attention should affect slot representations
    gate = sigmoid([slots, f_attn] @ Wg)

6. MLP update (like a transformer FFN, just a nonlinear transformation)
    h = ReLU([slots, f_attn] @ Wf0)
    h = h @ Wf1

7. Final velocity
    dslots/dt = gate * f_attn + h

8. Decoder <--- original to locatellos paper
"""

class SlotODEFunc(eqx.Module):
    """Autonomous ODE vector field for slot dynamics.

    The vector field depends only on the slot state (and fixed scene features),
    not on time — giving an autonomous system amenable to fixed-point and
    stability analysis.
    """
    # dead fields preserved for TPU XLA compilation compatibility
    _unused_0: None
    _unused_1: None
    _unused_2: None
    _unused_3: None

    W_q: jax.Array
    W_gate: jax.Array
    W_ff0: jax.Array
    W_ff1: jax.Array

    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    scale: float

    slot_dim: int = eqx.field(static=True)
    mlp_hidden: int = eqx.field(static=True)
    autonomous: bool = eqx.field(static=True)

    def __init__(self, slot_dim: int, mlp_hidden: int = 128, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim
        self.mlp_hidden = mlp_hidden
        self.autonomous = True

        self._unused_0 = None
        self._unused_1 = None
        self._unused_2 = None
        self._unused_3 = None

        self.W_q = jax.random.normal(k1, (slot_dim, slot_dim)) * (slot_dim ** -0.5)
        self.W_gate = jax.random.normal(k2, (slot_dim, 2 * slot_dim)) * ((2 * slot_dim) ** -0.5)
        self.W_ff0 = jax.random.normal(k3, (mlp_hidden, 2 * slot_dim)) * ((2 * slot_dim) ** -0.5)
        self.W_ff1 = jax.random.normal(k4, (slot_dim, mlp_hidden)) * (mlp_hidden ** -0.5)

        self.norm_attn = eqx.nn.LayerNorm(slot_dim)
        self.norm_ff = eqx.nn.LayerNorm(slot_dim)

    def __call__(self, t: float, slots: jax.Array, args: tuple) -> jax.Array:
        """
        slots: [B, N_slots, slot_dim]
        args: (k, v) — precomputed keys and values that remain fixed
          k: [B, N_feat, slot_dim]
          v: [B, N_feat, slot_dim]

        returns: velocity [B, N_slots, slot_dim]
        """
        k, v = args

        if self.autonomous:
            Wq, Wg, Wf0, Wf1 = self.W_q, self.W_gate, self.W_ff0, self.W_ff1

        slots_norm = jax.vmap(jax.vmap(self.norm_attn))(slots)
        q = jnp.einsum('bnd,od->bno', slots_norm, Wq)

        att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
        att = jax.nn.softmax(att_logits, axis=1)
        att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)
        f_attn = jnp.einsum('bnm,bmd->bnd', att, v)

        gate_input = jnp.concatenate([slots_norm, f_attn], axis=-1)
        gate = jax.nn.sigmoid(jnp.einsum('bnd,od->bno', gate_input, Wg))

        # residual MLP
        slots_ff = jax.vmap(jax.vmap(self.norm_ff))(slots)
        ff_input = jnp.concatenate([slots_ff, f_attn], axis=-1)
        h = jnp.einsum('bnd,od->bno', ff_input, Wf0)
        h = jax.nn.relu(h)
        h = jnp.einsum('bnd,od->bno', h, Wf1)

        return gate * f_attn + h

# slot attention as a continuous-time neural dynamical system
class SlotAttentionODE(eqx.Module):
    # hyperparameters
    num_slots: int
    slot_dim: int
    T: float # integration time
    dt0: float = eqx.field(static=True)

    # learnable slot initialization
    slots_mu: jax.Array # [1, 1, slot_dim]
    slots_log_sigma: jax.Array # [1, 1, slot_dim]

    # input projection
    norm_input: eqx.nn.LayerNorm
    fc_input: eqx.nn.Linear

    # fixed K and V projections (the scene itself doesn't change during integration)
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear

    # ODE dynamics
    slot_ode_func: SlotODEFunc

    def __init__(self, num_slots: int, slot_dim: int, enc_dim: int, num_iter: int = 3, dt0: float = 1.0,
                 *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.T = float(num_iter)
        self.dt0 = dt0

        self.slots_mu = jax.random.normal(k1, (1, 1, slot_dim))
        self.slots_log_sigma = jnp.full((1, 1, slot_dim), math.log(2.0))

        self.norm_input = eqx.nn.LayerNorm(enc_dim)
        self.fc_input = eqx.nn.Linear(enc_dim, slot_dim, key=k2)

        self.to_k = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k3)
        self.to_v = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k4)

        self.slot_ode_func = SlotODEFunc(slot_dim, key=key)

    def initialize_slots(self, batch_size: int, key: jax.Array) -> jax.Array:
        mu = jnp.broadcast_to(self.slots_mu, (batch_size, self.num_slots, self.slot_dim))
        sigma = jnp.broadcast_to(jnp.exp(self.slots_log_sigma), (batch_size, self.num_slots, self.slot_dim))
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise # [B, N_slots, slot_dim]

    def __call__(self, enc_feat: jax.Array, key: jax.Array, return_traj: bool = False) -> tuple:
        B = enc_feat.shape[0] # enc_feat is of shape [B, N_feat, D_enc] where N_feat = H * W

        # project input features and precompute K and V
        feat_norm = jax.vmap(jax.vmap(self.norm_input))(enc_feat) # [B, N_feat, D_enc]
        feat = jax.vmap(jax.vmap(self.fc_input))(feat_norm) # [B, N_feat, D_slot]
        k = jax.vmap(jax.vmap(self.to_k))(feat) # [B, N_feat, D_slot]
        v = jax.vmap(jax.vmap(self.to_v))(feat) # [B, N_feat, D_slot]

        # initialize slots
        slots_0 = self.initialize_slots(B, key) # [B, N_slots, D_slot]

        # set up the ODE term with diffrax
        term = diffrax.ODETerm(self.slot_ode_func)
        solver = diffrax.Euler()
        stepsize_controller = diffrax.ConstantStepSize()

        if return_traj:
            # save at actual solver steps to avoid interpolation artifacts
            ts = jnp.arange(0.0, self.T + self.dt0, self.dt0)
            ts = jnp.clip(ts, 0.0, self.T)
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        n_steps = int(self.T / self.dt0)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=self.T, dt0=self.dt0,
            y0=slots_0, args=(k, v),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=n_steps + 16,
        )

        if return_traj:
            traj = sol.ys
            slots_final = traj[-1]
            return slots_final, traj
        else:
            slots_final = sol.ys[0]
            return slots_final

class SlotODEModel(eqx.Module):
    resolution: tuple
    num_slots: int

    enc: Encoder
    slot_attention_ode: SlotAttentionODE
    dec: SpatialBroadcastDecoder

    def __init__(self, resolution: tuple = (64, 64), num_slots: int = 7, slot_dim: int = 64, enc_hidden_dim: int = 64,
                 num_iter: int = 3, dt0: float = 1.0, *, key: jax.Array):
        k_enc, k_sa, k_dec = jax.random.split(key, 3)

        self.resolution = resolution
        self.num_slots = num_slots

        self.enc = Encoder(enc_hidden_dim, key=k_enc)

        self.slot_attention_ode = SlotAttentionODE(
            num_slots=num_slots, slot_dim=slot_dim, enc_dim=enc_hidden_dim,
            num_iter=num_iter, dt0=dt0, key=k_sa
        )

        self.dec = SpatialBroadcastDecoder(slot_dim, resolution, dec_hidden_dim=enc_hidden_dim, key=k_dec)

    def __call__(self, image: jax.Array, *, key: jax.Array, return_traj: bool = False) -> tuple:
        """
        image: [B, 3, H, W]
        key: PRNG key (needed for random slot initialization)

        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
            slots: [B, N_slots, D_slot]
            (if return_traj) traj: [T, B, N_slots, D_slot]
        """
        enc_feat = self.enc(image)

        if return_traj:
            slots, traj = self.slot_attention_ode(enc_feat, key, return_traj=True)
        else:
            slots = self.slot_attention_ode(enc_feat, key)
            traj = None

        recon, masks = self.dec(slots)

        if return_traj:
            return recon, masks, slots, traj
        else:
            return recon, masks, slots
