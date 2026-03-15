import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import math

# soft positional embedding
class SoftPositionalEmbedding(eqx.Module):
    
    dense: eqx.nn.Linear # [4 -> hidden_dim]

    def __init__(self, hidden_dim: int, *, key: jax.Array):
        self.dense = eqx.nn.Linear(4, hidden_dim, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: [B, C, H, W]  (NCHW)
        returns: [B, C, H, W]
        """
        _, _, H, W = x.shape

        # create normalized position grid in [0, 1] (distance from borders)
        grid_h = jnp.linspace(0, 1, H)
        grid_w = jnp.linspace(0, 1, W)
        yy, xx = jnp.meshgrid(grid_h, grid_w, indexing='ij')

        # [H, W, 4] — distances from all 4 edges
        pos_grid = jnp.stack([xx, yy, 1.0 - xx, 1.0 - yy], axis=-1)

        # project: [H, W, 4] -> [H, W, C]
        # eqx.nn.Linear expects input shape [..., in_features]
        # so we vmap over spatial dims or just let broadcasting handle it
        pos_embed = jax.vmap(jax.vmap(self.dense))(pos_grid)  # [H, W, C]

        # -> [C, H, W] to match input layout
        pos_embed = jnp.transpose(pos_embed, (2, 0, 1))  # [C, H, W]

        # broadcast add over batch dimension
        return x + pos_embed[None]  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Time-dependent weight generation (DiffEqFormer-style hypernetwork)
# ---------------------------------------------------------------------------

class TimeDepWeight(eqx.Module):
    """Generates a weight matrix W(t) as a function of continuous time t.

    Architecture (following DiffEqFormer, Tong et al. 2025):
        W(t) = Proj(MLP(Sinusoidal(t)))

    - Sinusoidal: scalar t -> Concat(t, sin(w*t), cos(w*t)) with log-spaced
      frequencies w, giving a rich multi-scale time representation.
    - MLP: 2-layer network with SiLU activation.
    - Proj: linear projection reshaped to (d_out, d_in) weight matrix.

    Each weight matrix (Q, MLP_0, MLP_1) gets its own MLP+Proj head but
    shares the same sinusoidal frequency vector (following Figure 17 left).
    """
    # fixed sinusoidal frequencies (not trainable)
    freqs: jax.Array  # [n_freq]

    # MLP: sinusoidal_dim -> d_emb -> d_emb
    mlp_0: eqx.nn.Linear
    mlp_1: eqx.nn.Linear

    # projection: d_emb -> d_out * d_in
    proj: eqx.nn.Linear

    d_in: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_out: int, d_emb: int = 64,
                 n_freq: int = 128, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)

        self.d_in = d_in
        self.d_out = d_out

        # log-spaced frequencies: w_i = -log(10^4) * i / n_freq
        self.freqs = -math.log(1e4) * jnp.arange(n_freq) / n_freq

        # sinusoidal input dim: t + sin(w*t) + cos(w*t) = 1 + 2*n_freq
        sin_dim = 1 + 2 * n_freq

        # 2-layer MLP with SiLU
        self.mlp_0 = eqx.nn.Linear(sin_dim, d_emb, key=k1)
        self.mlp_1 = eqx.nn.Linear(d_emb, d_emb, key=k2)

        # project to flattened weight matrix
        self.proj = eqx.nn.Linear(d_emb, d_out * d_in, key=k3)

    def __call__(self, t: float) -> jax.Array:
        """Map scalar t -> weight matrix [d_out, d_in]."""
        # sinusoidal embedding
        wt = self.freqs * t  # [n_freq]
        emb = jnp.concatenate([jnp.array([t]), jnp.sin(wt), jnp.cos(wt)])

        # MLP with SiLU activation
        h = self.mlp_0(emb)
        h = jax.nn.silu(h)
        h = self.mlp_1(h)
        h = jax.nn.silu(h)

        # project and reshape to weight matrix
        w = self.proj(h)
        return w.reshape(self.d_out, self.d_in)


# ---------------------------------------------------------------------------
# SlotODE dynamics function (non-autonomous, time-dependent weights)
# ---------------------------------------------------------------------------

# class SlotODEFunc(eqx.Module):
#     """Non-autonomous ODE vector field: ds/dt = gate * f(s, features, t) + g(s, t).

#     DiffEqFormer-style formulation (Tong et al. 2025, Eq. 2) adapted for
#     Slot Attention, with per-dimension gating on the attention signal.

#     The gate sees both the current slot state and the proposed attention update,
#     and learns to modulate each dimension of the attention velocity independently.
#     This provides the selective per-dimension integration that the GRU gives
#     baseline slot attention, while remaining compatible with ODE integration.

#         f = cross-attention (slots query features, softmax over slots)
#         gate = sigmoid(W_gate(t) @ [slots_norm; f_attn])
#         g = feedforward MLP refinement

#     velocity = gate * f_attn + g_ff
#     """
#     # time-dependent weight generators
#     tdw_q: TimeDepWeight       # generates Q projection    [slot_dim, slot_dim]
#     tdw_gate: TimeDepWeight    # generates gate projection [slot_dim, 2*slot_dim]
#     tdw_ff0: TimeDepWeight     # generates FF layer 0      [mlp_hidden, slot_dim]
#     tdw_ff1: TimeDepWeight     # generates FF layer 1      [slot_dim, mlp_hidden]

#     norm_attn: eqx.nn.LayerNorm
#     norm_ff: eqx.nn.LayerNorm
#     scale: float

#     slot_dim: int = eqx.field(static=True)
#     mlp_hidden: int = eqx.field(static=True)

#     def __init__(self, slot_dim: int, mlp_hidden: int = 128,
#                  d_emb: int = 64, *, key: jax.Array):
#         k1, k2, k3, k4 = jax.random.split(key, 4)
#         self.scale = slot_dim ** -0.5
#         self.slot_dim = slot_dim
#         self.mlp_hidden = mlp_hidden

#         self.tdw_q = TimeDepWeight(slot_dim, slot_dim, d_emb=d_emb, key=k1)
#         self.tdw_gate = TimeDepWeight(2 * slot_dim, slot_dim, d_emb=d_emb, key=k2)
#         self.tdw_ff0 = TimeDepWeight(slot_dim, mlp_hidden, d_emb=d_emb, key=k3)
#         self.tdw_ff1 = TimeDepWeight(mlp_hidden, slot_dim, d_emb=d_emb, key=k4)

#         self.norm_attn = eqx.nn.LayerNorm(slot_dim)
#         self.norm_ff = eqx.nn.LayerNorm(slot_dim)

#     def __call__(self, t: float, slots: jax.Array, args: tuple) -> jax.Array:
#         """
#         slots:  [B, N_slots, D]
#         args:   (k, v) — static precomputed keys and values
#           k: [B, N_feat, D]
#           v: [B, N_feat, D]

#         returns: velocity [B, N_slots, D]
#         """
#         k, v = args

#         # time-dependent weights
#         W_q = self.tdw_q(t)        # [D, D]
#         W_gate = self.tdw_gate(t)  # [D, 2*D]
#         W_ff0 = self.tdw_ff0(t)    # [mlp_hidden, D]
#         W_ff1 = self.tdw_ff1(t)    # [D, mlp_hidden]

#         # --- f: cross-attention ---
#         slots_norm = jax.vmap(jax.vmap(self.norm_attn))(slots)
#         q = jnp.einsum('bnd,od->bno', slots_norm, W_q)

#         att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
#         att = jax.nn.softmax(att_logits, axis=1)
#         att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)
#         f_attn = jnp.einsum('bnm,bmd->bnd', att, v)

#         # --- gate: per-dimension modulation of attention signal ---
#         gate_input = jnp.concatenate([slots_norm, f_attn], axis=-1)  # [B, N, 2*D]
#         gate = jax.nn.sigmoid(
#             jnp.einsum('bnd,od->bno', gate_input, W_gate)
#         )  # [B, N, D]

#         # --- g: feedforward ---
#         slots_ff = jax.vmap(jax.vmap(self.norm_ff))(slots)
#         h = jnp.einsum('bnd,od->bno', slots_ff, W_ff0)
#         h = jax.nn.relu(h)
#         h = jnp.einsum('bnd,od->bno', h, W_ff1)
#         g_ff = h

#         # velocity = gated attention + feedforward
#         return gate * f_attn + g_ff
    
class SlotODEFunc(eqx.Module):
    """Non-autonomous ODE vector field: ds/dt = gate * f(s, features, t) + g(s, f_attn, t).
 
    Two changes from baseline SlotODEFunc:
 
    1. Per-dimension gate on the attention signal (Option 2):
       gate = sigmoid(W_gate(t) @ [slots_norm; f_attn])
       Allows selective per-dimension acceptance of attention updates.
 
    2. Informed feedforward (Option 1):
       The feedforward MLP receives both the slot state AND the attention
       output, so it can learn corrections based on what attention proposed.
       e.g. "attention pulled toward two objects, push back toward one"
 
    velocity = gate * f_attn + g_ff(slots, f_attn)
    """
    # time-dependent weight generators
    tdw_q: TimeDepWeight       # generates Q projection    [slot_dim, slot_dim]
    tdw_gate: TimeDepWeight    # generates gate projection [slot_dim, 2*slot_dim]
    tdw_ff0: TimeDepWeight     # generates FF layer 0      [mlp_hidden, 2*slot_dim]  <- changed
    tdw_ff1: TimeDepWeight     # generates FF layer 1      [slot_dim, mlp_hidden]
 
    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    scale: float
 
    slot_dim: int = eqx.field(static=True)
    mlp_hidden: int = eqx.field(static=True)
 
    def __init__(self, slot_dim: int, mlp_hidden: int = 128,
                 d_emb: int = 64, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim
        self.mlp_hidden = mlp_hidden
 
        self.tdw_q = TimeDepWeight(slot_dim, slot_dim, d_emb=d_emb, key=k1)
        self.tdw_gate = TimeDepWeight(2 * slot_dim, slot_dim, d_emb=d_emb, key=k2)
        self.tdw_ff0 = TimeDepWeight(2 * slot_dim, mlp_hidden, d_emb=d_emb, key=k3)  # 2*D input
        self.tdw_ff1 = TimeDepWeight(mlp_hidden, slot_dim, d_emb=d_emb, key=k4)
 
        self.norm_attn = eqx.nn.LayerNorm(slot_dim)
        self.norm_ff = eqx.nn.LayerNorm(slot_dim)
 
    def __call__(self, t: float, slots: jax.Array, args: tuple) -> jax.Array:
        """
        slots:  [B, N_slots, D]
        args:   (k, v) — static precomputed keys and values
          k: [B, N_feat, D]
          v: [B, N_feat, D]
 
        returns: velocity [B, N_slots, D]
        """
        k, v = args
 
        # time-dependent weights
        W_q = self.tdw_q(t)        # [D, D]
        W_gate = self.tdw_gate(t)  # [D, 2*D]
        W_ff0 = self.tdw_ff0(t)    # [mlp_hidden, 2*D]
        W_ff1 = self.tdw_ff1(t)    # [D, mlp_hidden]
 
        # --- f: cross-attention ---
        slots_norm = jax.vmap(jax.vmap(self.norm_attn))(slots)
        q = jnp.einsum('bnd,od->bno', slots_norm, W_q)
 
        att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
        att = jax.nn.softmax(att_logits, axis=1)
        att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)
        f_attn = jnp.einsum('bnm,bmd->bnd', att, v)
 
        # --- gate: per-dimension modulation of attention signal ---
        gate_input = jnp.concatenate([slots_norm, f_attn], axis=-1)  # [B, N, 2*D]
        gate = jax.nn.sigmoid(
            jnp.einsum('bnd,od->bno', gate_input, W_gate)
        )  # [B, N, D]
 
        # --- g: informed feedforward (sees both slot state and attention output) ---
        slots_ff = jax.vmap(jax.vmap(self.norm_ff))(slots)
        ff_input = jnp.concatenate([slots_ff, f_attn], axis=-1)      # [B, N, 2*D]
        h = jnp.einsum('bnd,od->bno', ff_input, W_ff0)               # [B, N, mlp_hidden]
        h = jax.nn.relu(h)
        h = jnp.einsum('bnd,od->bno', h, W_ff1)                      # [B, N, D]
        g_ff = h
 
        # velocity = gated attention + informed feedforward
        return gate * f_attn + g_ff

# ---------------------------------------------------------------------------
# Slot Attention ODE module
# ---------------------------------------------------------------------------

class SlotAttentionODE(eqx.Module):
    """Initializes slots and integrates the ODE to evolve them.

    Diffrax usage:
      - ODETerm wraps our vector field (SlotODEFunc)
      - We use Tsit5 with ConstantStepSize for training (stable, predictable)
      - For analysis, swap to Dopri5 + PIDController for adaptive stepping
      - SaveAt(ts=...) captures the full trajectory for visualization
    """
    num_slots: int
    slot_dim: int
    T: float
    solver_name: str = eqx.field(static=True)  # not a trainable param
    dt0: float = eqx.field(static=True)        # not a trainable param

    # learnable per-slot initialization parameters
    slots_mu: jax.Array         # [1, num_slots, slot_dim]
    slots_log_sigma: jax.Array  # [1, 1, slot_dim]

    # input projection (LayerNorm -> Linear to slot_dim)
    norm_input: eqx.nn.LayerNorm
    fc_input: eqx.nn.Linear

    # static K, V projections (scene description doesn't change during integration)
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear

    # ODE dynamics (time-dependent Q and FF only)
    slot_ode_func: SlotODEFunc

    def __init__(self, num_slots: int, slot_dim: int, enc_dim: int,
                 num_iter: int = 3, solver: str = "tsit5", *,
                 key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.T = float(num_iter)
        self.solver_name = solver

        # step sizes: Euler needs larger steps (1.0), higher-order solvers use 0.25
        self.dt0 = 1.0 if solver == "euler" else 0.25

        # learnable per-slot init — each slot gets a distinct learned mean
        self.slots_mu = jax.random.normal(k1, (1, num_slots, slot_dim)) * 0.1
        self.slots_log_sigma = jnp.zeros((1, 1, slot_dim))

        # input projection (maps encoder features to slot_dim)
        self.norm_input = eqx.nn.LayerNorm(enc_dim)
        self.fc_input = eqx.nn.Linear(enc_dim, slot_dim, key=k2)

        # static K, V projections (precomputed once per forward pass)
        self.to_k = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k3)
        self.to_v = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k4)

        # ODE dynamics
        self.slot_ode_func = SlotODEFunc(slot_dim, key=key)

    def initialize_slots(self, batch_size: int, key: jax.Array) -> jax.Array:
        """Sample initial slots from learned Gaussian. [B, N_slots, D]"""
        mu = jnp.broadcast_to(self.slots_mu, (batch_size, self.num_slots, self.slot_dim))
        sigma = jnp.broadcast_to(
            jnp.exp(self.slots_log_sigma),
            (batch_size, self.num_slots, self.slot_dim)
        )
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise

    def __call__(self, enc_feat: jax.Array, key: jax.Array,
                 return_traj: bool = False, num_traj_pts: int = 20) -> tuple:
        """
        enc_feat: [B, N_feat, D_encoder]
        key: PRNG key for slot initialization
        return_traj: if True, return trajectory at multiple time points

        returns:
            slots: [B, N_slots, D_slot]
            (optional) traj: [num_traj_pts, B, N_slots, D_slot]
        """
        B = enc_feat.shape[0]

        # project input features and precompute static K, V
        feat_norm = jax.vmap(jax.vmap(self.norm_input))(enc_feat)  # [B, N_feat, D_enc]
        feat = jax.vmap(jax.vmap(self.fc_input))(feat_norm)        # [B, N_feat, D_slot]
        k = jax.vmap(jax.vmap(self.to_k))(feat)                   # [B, N_feat, D_slot]
        v = jax.vmap(jax.vmap(self.to_v))(feat)                   # [B, N_feat, D_slot]

        # initialize slots
        slots_0 = self.initialize_slots(B, key)  # [B, N_slots, D_slot]

        # set up diffrax ODE solve
        term = diffrax.ODETerm(self.slot_ode_func)

        if self.solver_name == "euler":
            solver = diffrax.Euler()
            stepsize_controller = diffrax.ConstantStepSize()
        elif self.solver_name == "tsit5":
            solver = diffrax.Tsit5()
            stepsize_controller = diffrax.ConstantStepSize()
        elif self.solver_name == "dopri5":
            solver = diffrax.Dopri5()
            stepsize_controller = diffrax.PIDController(atol=1e-5, rtol=1e-5)
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

        if return_traj:
            ts = jnp.linspace(0.0, self.T, num_traj_pts)
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=self.T,
            dt0=self.dt0,
            y0=slots_0,
            args=(k, v),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )

        if return_traj:
            # sol.ys: [num_traj_pts, B, N_slots, D_slot]
            traj = sol.ys
            slots_final = traj[-1]
            return slots_final, traj
        else:
            # sol.ys: [1, B, N_slots, D_slot] (only t1 saved)
            slots_final = sol.ys[0]
            return slots_final


# ---------------------------------------------------------------------------
# Spatial Broadcast Decoder
# ---------------------------------------------------------------------------

class SpatialBroadcastDecoder(eqx.Module):
    """Decodes each slot independently into RGB + mask via spatial broadcast.

    Matches Locatello et al. 2020 (Table 6): slot vector is broadcast onto a
    small 8x8 grid, positional embeddings added, then upsampled to full
    resolution via transposed convolutions (stride 2). Slot masks are
    softmax-normalized (competition for pixels).

    For 64x64: broadcast 8x8 -> 3 deconv stride-2 (8->16->32->64) + 1 conv + output.
    """
    resolution: tuple
    broadcast_size: tuple
    pos_embed: SoftPositionalEmbedding

    # deconv layers (stride 2, upsampling)
    deconv0: eqx.nn.ConvTranspose2d
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d

    # regular conv + output
    conv0: eqx.nn.Conv2d
    conv_out: eqx.nn.Conv2d

    def __init__(self, slot_dim: int, resolution: tuple = (64, 64),
                 dec_hidden_dim: int = 64, *, key: jax.Array):
        k0, k1, k2, k3, k4, kp = jax.random.split(key, 6)

        self.resolution = resolution
        self.broadcast_size = (8, 8)
        self.pos_embed = SoftPositionalEmbedding(slot_dim, key=kp)

        # deconv stride-2 layers: 8 -> 16 -> 32 -> 64
        self.deconv0 = eqx.nn.ConvTranspose2d(
            slot_dim, dec_hidden_dim, kernel_size=5, stride=2,
            padding=2, output_padding=1, key=k0)
        self.deconv1 = eqx.nn.ConvTranspose2d(
            dec_hidden_dim, dec_hidden_dim, kernel_size=5, stride=2,
            padding=2, output_padding=1, key=k1)
        self.deconv2 = eqx.nn.ConvTranspose2d(
            dec_hidden_dim, dec_hidden_dim, kernel_size=5, stride=2,
            padding=2, output_padding=1, key=k2)

        # stride-1 conv + output
        self.conv0 = eqx.nn.Conv2d(dec_hidden_dim, dec_hidden_dim, 5, padding=2, key=k3)
        self.conv_out = eqx.nn.Conv2d(dec_hidden_dim, 4, 3, padding=1, key=k4)  # 3 RGB + 1 mask

    def decode_single(self, slot: jax.Array) -> jax.Array:
        """Decode a single slot [D] -> [4, H, W] (3 RGB + 1 mask logit)."""
        bH, bW = self.broadcast_size

        # spatial broadcast to small grid: [D] -> [D, bH, bW]
        x = jnp.broadcast_to(slot[:, None, None], (slot.shape[0], bH, bW))

        # add positional embedding at broadcast resolution
        x = self.pos_embed(x[None])[0]  # [C, bH, bW]

        # deconv upsample: 8 -> 16 -> 32 -> 64
        x = jax.nn.relu(self.deconv0(x))
        x = jax.nn.relu(self.deconv1(x))
        x = jax.nn.relu(self.deconv2(x))

        # stride-1 conv + output
        x = jax.nn.relu(self.conv0(x))
        x = self.conv_out(x)  # [4, H, W]

        return x

    def __call__(self, slots: jax.Array) -> tuple:
        """
        slots: [B, N_slots, D_slot]
        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
        """
        B, N_slots, D = slots.shape

        # decode each slot independently
        # vmap over batch, then over slots
        decode_batch_slots = jax.vmap(jax.vmap(self.decode_single))
        x = decode_batch_slots(slots)  # [B, N_slots, 4, H, W]

        recons = x[:, :, :3, :, :]          # [B, N_slots, 3, H, W]
        mask_logits = x[:, :, 3, :, :]      # [B, N_slots, H, W]

        # softmax over slots for each pixel (competition)
        masks = jax.nn.softmax(mask_logits, axis=1)  # [B, N_slots, H, W]

        # mixture reconstruction
        recon = (recons * masks[:, :, None, :, :]).sum(axis=1)  # [B, 3, H, W]

        return recon, masks


# ---------------------------------------------------------------------------
# Full SlotODE Model
# ---------------------------------------------------------------------------

class SlotODEModel(eqx.Module):
    """Complete SlotODE autoencoder.

    Usage:
        key = jax.random.key(0)
        model = SlotODEModel(key=key)

        # forward pass (need a fresh key for slot initialization randomness)
        key, subkey = jax.random.split(key)
        recon, masks, slots = model(images, key=subkey)

        # JIT-compiled forward:
        @eqx.filter_jit
        def forward(model, images, key):
            return model(images, key=key)
    """
    resolution: tuple
    num_slots: int

    # CNN encoder
    enc_conv0: eqx.nn.Conv2d
    enc_conv1: eqx.nn.Conv2d
    enc_conv2: eqx.nn.Conv2d
    enc_conv3: eqx.nn.Conv2d

    enc_pos: SoftPositionalEmbedding

    # encoder MLP (residual)
    enc_norm: eqx.nn.LayerNorm
    enc_fc0: eqx.nn.Linear
    enc_fc1: eqx.nn.Linear

    # slot attention ODE
    slot_attention_ode: SlotAttentionODE

    # decoder
    dec: SpatialBroadcastDecoder

    def __init__(self, resolution: tuple = (64, 64), num_slots: int = 7,
                 slot_dim: int = 64, enc_hidden_dim: int = 64,
                 num_iter: int = 3, solver: str = "tsit5", *, key: jax.Array):
        (k_c0, k_c1, k_c2, k_c3, k_pos, k_f0, k_f1,
         k_sa, k_dec) = jax.random.split(key, 9)

        self.resolution = resolution
        self.num_slots = num_slots

        # CNN encoder (identical architecture to PyTorch version)
        self.enc_conv0 = eqx.nn.Conv2d(3, enc_hidden_dim, 5, padding=2, key=k_c0)
        self.enc_conv1 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c1)
        self.enc_conv2 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c2)
        self.enc_conv3 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c3)

        self.enc_pos = SoftPositionalEmbedding(enc_hidden_dim, key=k_pos)

        # encoder MLP
        self.enc_norm = eqx.nn.LayerNorm(enc_hidden_dim)
        self.enc_fc0 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f0)
        self.enc_fc1 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f1)

        # slot attention ODE
        self.slot_attention_ode = SlotAttentionODE(
            num_slots=num_slots, slot_dim=slot_dim, enc_dim=enc_hidden_dim,
            num_iter=num_iter, solver=solver, key=k_sa
        )

        # decoder
        self.dec = SpatialBroadcastDecoder(
            slot_dim, resolution, dec_hidden_dim=enc_hidden_dim, key=k_dec
        )

    def encode(self, image: jax.Array) -> jax.Array:
        """
        image: [B, 3, H, W]
        returns: [B, H*W, enc_hidden_dim]
        """
        # vmap encoder CNN over batch
        def encode_single(img):
            # img: [3, H, W]
            x = jax.nn.relu(self.enc_conv0(img))
            x = jax.nn.relu(self.enc_conv1(x))
            x = jax.nn.relu(self.enc_conv2(x))
            x = jax.nn.relu(self.enc_conv3(x))
            return x  # [C, H, W]

        x = jax.vmap(encode_single)(image)  # [B, C, H, W]

        # add positional embeddings
        x = self.enc_pos(x)  # [B, C, H, W]

        # flatten to tokens: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(0, 2, 1)  # [B, H*W, C]

        # residual MLP
        h = jax.vmap(jax.vmap(self.enc_norm))(x)
        h = jax.vmap(jax.vmap(self.enc_fc0))(h)
        h = jax.nn.relu(h)
        h = jax.vmap(jax.vmap(self.enc_fc1))(h)
        enc_feat = x + h  # residual connection

        return enc_feat  # [B, H*W, C]

    def __call__(self, image: jax.Array, *, key: jax.Array,
                 return_traj: bool = False) -> tuple:
        """
        image: [B, 3, H, W]
        key: PRNG key (needed for random slot initialization)

        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
            slots: [B, N_slots, D_slot]
            (if return_traj) traj: [T, B, N_slots, D_slot]
        """
        enc_feat = self.encode(image)  # [B, H*W, C]

        # evolve slots via ODE
        if return_traj:
            slots, traj = self.slot_attention_ode(
                enc_feat, key, return_traj=True
            )
        else:
            slots = self.slot_attention_ode(enc_feat, key)
            traj = None

        # decode
        recon, masks = self.dec(slots)

        if return_traj:
            return recon, masks, slots, traj
        else:
            return recon, masks, slots
