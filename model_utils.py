import jax
import jax.numpy as jnp
import equinox as eqx

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

# CNN encoder
class Encoder(eqx.Module):
    # convolutional layers
    conv0: eqx.nn.Conv2d
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    pos: SoftPositionalEmbedding

    # dense layers
    norm: eqx.nn.LayerNorm
    fc0: eqx.nn.Linear
    fc1: eqx.nn.Linear

    def __init__(self, enc_hidden_dim: int = 64, *, key: jax.Array):
        k_c0, k_c1, k_c2, k_c3, k_pos, k_f0, k_f1 = jax.random.split(key, 7)

        self.conv0 = eqx.nn.Conv2d(3, enc_hidden_dim, 5, padding=2, key=k_c0)
        self.conv1 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c1)
        self.conv2 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c2)
        self.conv3 = eqx.nn.Conv2d(enc_hidden_dim, enc_hidden_dim, 5, padding=2, key=k_c3)

        self.pos = SoftPositionalEmbedding(enc_hidden_dim, key=k_pos)

        self.norm = eqx.nn.LayerNorm(enc_hidden_dim)
        self.fc0 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f0)
        self.fc1 = eqx.nn.Linear(enc_hidden_dim, enc_hidden_dim, key=k_f1)

    def __call__(self, image: jax.Array) -> jax.Array:
        """
        image: [B, 3, H, W]
        returns: [B, H*W, enc_hidden_dim]
        """
        def encode_single(img):
            x = jax.nn.relu(self.conv0(img))
            x = jax.nn.relu(self.conv1(x))
            x = jax.nn.relu(self.conv2(x))
            x = jax.nn.relu(self.conv3(x))
            return x # [C, H, W]

        x = jax.vmap(encode_single)(image) # [B, C, H, W]
        x = self.pos(x) # [B, C, H, W]

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(0, 2, 1) # [B, H * W, C]

        # residual MLP
        h = jax.vmap(jax.vmap(self.norm))(x)
        h = jax.vmap(jax.vmap(self.fc0))(h)
        h = jax.nn.relu(h)
        h = jax.vmap(jax.vmap(self.fc1))(h)
        return x + h # [B, H * W, C]

# spatial broadcast decoder for reconstruction
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

    # deconvolutional layers (upsampling with stride 2)
    deconv0: eqx.nn.ConvTranspose2d
    deconv1: eqx.nn.ConvTranspose2d
    deconv2: eqx.nn.ConvTranspose2d

    # convolutional layers (output)
    conv0: eqx.nn.Conv2d
    conv_out: eqx.nn.Conv2d

    def __init__(self, slot_dim: int, resolution: tuple = (64, 64), dec_hidden_dim: int = 64, *, key: jax.Array):
        k0, k1, k2, k3, k4, kp = jax.random.split(key, 6)

        self.resolution = resolution
        self.broadcast_size = (8, 8)
        self.pos_embed = SoftPositionalEmbedding(slot_dim, key=kp)

        self.deconv0 = eqx.nn.ConvTranspose2d(slot_dim, dec_hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1, key=k0)
        self.deconv1 = eqx.nn.ConvTranspose2d(dec_hidden_dim, dec_hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1, key=k1)
        self.deconv2 = eqx.nn.ConvTranspose2d(dec_hidden_dim, dec_hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1, key=k2)

        self.conv0 = eqx.nn.Conv2d(dec_hidden_dim, dec_hidden_dim, 5, padding=2, key=k3)
        self.conv_out = eqx.nn.Conv2d(dec_hidden_dim, 4, 3, padding=1, key=k4) # 3 output channels (RGB) + 1 mask

    def decode_single(self, slot: jax.Array) -> jax.Array:
        """Decode a single slot [D] -> [4, H, W] (3 RGB + 1 mask logit)."""
        bH, bW = self.broadcast_size

        # spatial broadcasting to a small grid, [D] -> [D, bH, bW]
        x = jnp.broadcast_to(slot[:, None, None], (slot.shape[0], bH, bW))

        # add positional embeddings at the broadcast resolution
        x = self.pos_embed(x[None])[0] # [C, bH, bW]

        # deconvolutions, 8 -> 16 -> 32 -> 64
        x = jax.nn.relu(self.deconv0(x))
        x = jax.nn.relu(self.deconv1(x))
        x = jax.nn.relu(self.deconv2(x))

        # stride-1 conv + output
        x = jax.nn.relu(self.conv0(x))
        x = self.conv_out(x)

        return x # [4, H, W]

    def __call__(self, slots: jax.Array) -> tuple:
        """
        slots: [B, N_slots, D_slot]
        returns:
            recon: [B, 3, H, W]
            masks: [B, N_slots, H, W]
        """
        B, N_slots, D = slots.shape

        # --- CHANGED: use eqx.filter_vmap so decoder weights are dynamic leaves,
        #     allowing dtype casts (e.g. float16 on GPU) to propagate through.
        #     Original used jax.vmap which captured self in a static closure,
        #     preventing tree-level float16 casts from reaching conv weights.
        # --- ORIGINAL (revert to this if needed):
        # decode_batch_slots = jax.vmap(jax.vmap(self.decode_single))
        # x = decode_batch_slots(slots) # [B, N_slots, 4, H, W]
        # --- NEW:
        decode_batch_slots = eqx.filter_vmap(eqx.filter_vmap(
            lambda dec, s: dec.decode_single(s),
            in_axes=(None, 0)), in_axes=(None, 0))
        x = decode_batch_slots(self, slots) # [B, N_slots, 4, H, W]

        recons = x[:, :, :3, :, :] # [B, N_slots, 3, H, W]
        mask_logits = x[:, :, 3, :, :] # [B, N_slots, H, W]

        # softmax over slots for each pixel
        masks = jax.nn.softmax(mask_logits, axis=1) # [B, N_slots, H, W]

        # mixture reconstruction
        recon = (recons * masks[:, :, None, :, :]).sum(axis=1) # [B, 3, H, W]

        return recon, masks