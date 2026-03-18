import math
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from model_utils import Encoder, SpatialBroadcastDecoder

# hypernetwork
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
    freqs: jax.Array # [n_freq]

    # MLP, sinusoidal_dim -> d_emb -> d_emb
    mlp_0: eqx.nn.Linear
    mlp_1: eqx.nn.Linear

    # projection, d_emb -> d_out * d_in
    proj: eqx.nn.Linear

    d_in: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_out: int, d_emb: int = 32, n_freq: int = 16, *, key: jax.Array):
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

class SlotODEFunc(eqx.Module):
    """ODE vector field for slot dynamics.

    When autonomous=False (default): time-dependent weights via hypernetworks.
    When autonomous=True: fixed learned weight matrices — the vector field
    depends only on the slot state, giving fixed-point convergence guarantees.
    """
    # time-dependent weight generators (used when autonomous=False)
    tdw_q: TimeDepWeight | None
    tdw_gate: TimeDepWeight | None
    tdw_ff0: TimeDepWeight | None
    tdw_ff1: TimeDepWeight | None

    # fixed weight matrices (used when autonomous=True)
    W_q: jax.Array | None
    W_gate: jax.Array | None
    W_ff0: jax.Array | None
    W_ff1: jax.Array | None

    norm_attn: eqx.nn.LayerNorm
    norm_ff: eqx.nn.LayerNorm
    scale: float

    slot_dim: int = eqx.field(static=True)
    mlp_hidden: int = eqx.field(static=True)
    autonomous: bool = eqx.field(static=True)

    def __init__(self, slot_dim: int, mlp_hidden: int = 128, d_emb: int = 32, n_freq: int = 16,
                 autonomous: bool = False, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim
        self.mlp_hidden = mlp_hidden
        self.autonomous = autonomous

        if autonomous:
            self.tdw_q = self.tdw_gate = self.tdw_ff0 = self.tdw_ff1 = None
            scale_q = (slot_dim ** -0.5)
            scale_gate = ((2 * slot_dim) ** -0.5)
            scale_ff0 = ((2 * slot_dim) ** -0.5)
            scale_ff1 = (mlp_hidden ** -0.5)
            self.W_q = jax.random.normal(k1, (slot_dim, slot_dim)) * scale_q
            self.W_gate = jax.random.normal(k2, (slot_dim, 2 * slot_dim)) * scale_gate
            self.W_ff0 = jax.random.normal(k3, (mlp_hidden, 2 * slot_dim)) * scale_ff0
            self.W_ff1 = jax.random.normal(k4, (slot_dim, mlp_hidden)) * scale_ff1
        else:
            self.W_q = self.W_gate = self.W_ff0 = self.W_ff1 = None
            self.tdw_q = TimeDepWeight(slot_dim, slot_dim, d_emb=d_emb, n_freq=n_freq, key=k1)
            self.tdw_gate = TimeDepWeight(2 * slot_dim, slot_dim, d_emb=d_emb, n_freq=n_freq, key=k2)
            self.tdw_ff0 = TimeDepWeight(2 * slot_dim, mlp_hidden, d_emb=d_emb, n_freq=n_freq, key=k3)
            self.tdw_ff1 = TimeDepWeight(mlp_hidden, slot_dim, d_emb=d_emb, n_freq=n_freq, key=k4)

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
        else:
            Wq = self.tdw_q(t)
            Wg = self.tdw_gate(t)
            Wf0 = self.tdw_ff0(t)
            Wf1 = self.tdw_ff1(t)

        slots_norm = jax.vmap(jax.vmap(self.norm_attn))(slots)
        q = jnp.einsum('bnd,od->bno', slots_norm, Wq)

        att_logits = jnp.einsum('bnd,bmd->bnm', q, k) * self.scale
        att = jax.nn.softmax(att_logits, axis=1)
        att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)
        f_attn = jnp.einsum('bnm,bmd->bnd', att, v)

        gate_input = jnp.concatenate([slots_norm, f_attn], axis=-1)
        gate = jax.nn.sigmoid(jnp.einsum('bnd,od->bno', gate_input, Wg))

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
    solver_name: str = eqx.field(static=True)
    dt0: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    # learnable slot initialization
    slots_mu: jax.Array # [1, num_slots, slot_dim]
    slots_log_sigma: jax.Array # [1, 1, slot_dim]

    # input projection
    norm_input: eqx.nn.LayerNorm
    fc_input: eqx.nn.Linear

    # fixed K and V projections (the scene itself doesn't change during integration)
    to_k: eqx.nn.Linear
    to_v: eqx.nn.Linear

    # ODE/SDE dynamics (Q(t), feedforward network, and noise coefficient)
    slot_ode_func: SlotODEFunc

    def __init__(self, num_slots: int, slot_dim: int, enc_dim: int, num_iter: int = 3, solver: str = "euler", dt0: float = 1.0,
                 d_emb: int = 32, n_freq: int = 16, autonomous: bool = False,
                 rtol: float = 1e-3, atol: float = 1e-6, max_steps: int = 256, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.T = float(num_iter)
        self.solver_name = solver
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

        self.slots_mu = jax.random.normal(k1, (1, 1, slot_dim))

        self.slots_log_sigma = jnp.full((1, 1, slot_dim), math.log(2.0))

        self.norm_input = eqx.nn.LayerNorm(enc_dim)
        self.fc_input = eqx.nn.Linear(enc_dim, slot_dim, key=k2)

        self.to_k = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k3)
        self.to_v = eqx.nn.Linear(slot_dim, slot_dim, use_bias=False, key=k4)

        self.slot_ode_func = SlotODEFunc(slot_dim, d_emb=d_emb, n_freq=n_freq, autonomous=autonomous, key=key)

    def initialize_slots(self, batch_size: int, key: jax.Array) -> jax.Array: # sample initial slots from a learned gaussian
        mu = jnp.broadcast_to(self.slots_mu, (batch_size, self.num_slots, self.slot_dim)) # [B, N_slots, slot_dim]
        sigma = jnp.broadcast_to(jnp.exp(self.slots_log_sigma), (batch_size, self.num_slots, self.slot_dim)) # [B, N_slots, slot_dim]
        noise = jax.random.normal(key, mu.shape)
        return mu + sigma * noise # [B, N_slots, slot_dim]

    def __call__(self, enc_feat: jax.Array, key: jax.Array, return_traj: bool = False, num_traj_pts: int = 20) -> tuple:
        """
        enc_feat: [B, N_feat, D_encoder]
        key: PRNG key for slot initialization
        return_traj: if True, return trajectory at multiple time points

        returns:
            slots: [B, N_slots, D_slot]
            (optional) traj: [num_traj_pts, B, N_slots, D_slot]
        """
        B = enc_feat.shape[0]

        # project input features and precompute K and V
        feat_norm = jax.vmap(jax.vmap(self.norm_input))(enc_feat) # [B, N_feat, D_enc]
        feat = jax.vmap(jax.vmap(self.fc_input))(feat_norm) # [B, N_feat, D_slot]
        k = jax.vmap(jax.vmap(self.to_k))(feat) # [B, N_feat, D_slot]
        v = jax.vmap(jax.vmap(self.to_v))(feat) # [B, N_feat, D_slot]

        # initialize slots
        slots_0 = self.initialize_slots(B, key) # [B, N_slots, D_slot]

        # set up the ODE term with diffrax
        term = diffrax.ODETerm(self.slot_ode_func)

        if self.solver_name == "euler":
            solver = diffrax.Euler()
            stepsize_controller = diffrax.ConstantStepSize()
        elif self.solver_name == "dopri5":
            solver = diffrax.Dopri5()
            stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        else:
            raise ValueError(f"Unknown solver: {self.solver_name}")

        if return_traj:
            saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, self.T, num_traj_pts))
        else:
            saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=self.T, dt0=self.dt0,
            y0=slots_0, args=(k, v),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=self.max_steps,
            throw=False,
        )

        if return_traj:
            traj = sol.ys # [num_traj_pts, B, N_slots, D_slot]
            slots_final = traj[-1]
            return slots_final, traj
        else:
            slots_final = sol.ys[0] # sol.ys, [1, B, N_slots, D_slot] (only save t1)
            return slots_final

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

    enc: Encoder
    slot_attention_ode: SlotAttentionODE
    dec: SpatialBroadcastDecoder

    def __init__(self, resolution: tuple = (64, 64), num_slots: int = 7, slot_dim: int = 64, enc_hidden_dim: int = 64,
                 num_iter: int = 3, solver: str = "euler", dt0: float = 1.0,
                 d_emb: int = 32, n_freq: int = 16, autonomous: bool = False,
                 rtol: float = 1e-3, atol: float = 1e-6, max_steps: int = 256, *, key: jax.Array):
        k_enc, k_sa, k_dec = jax.random.split(key, 3)

        self.resolution = resolution
        self.num_slots = num_slots

        self.enc = Encoder(enc_hidden_dim, key=k_enc)

        self.slot_attention_ode = SlotAttentionODE(
            num_slots=num_slots, slot_dim=slot_dim, enc_dim=enc_hidden_dim,
            num_iter=num_iter, solver=solver, dt0=dt0, d_emb=d_emb, n_freq=n_freq,
            autonomous=autonomous, rtol=rtol, atol=atol, max_steps=max_steps, key=k_sa
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
        enc_feat = self.enc(image) # [B, H * W, C]

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