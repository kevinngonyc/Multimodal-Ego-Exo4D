import torch
import torch.nn as nn
from TimeSformer.timesformer.models.vit import TimeSformer

class ProfVLM(nn.Module):
    """Encode videos per view, then fuse their features with the projector."""

    def __init__(self, pretrained_timesformer, img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time'):
        super().__init__()
        self.video_encoder = TimeSformer(img_size=img_size, num_classes=num_classes, num_frames=num_frames, attention_type=attention_type,  pretrained_model=pretrained_timesformer)
        self.agp = AttentiveGatedProjector()

    def forward(self, x):
        """Accept (B, V, C, T, H, W), or (B, C, T, H, W) for one view."""
        if x.ndim == 5:
            x = x.unsqueeze(1)
        if x.ndim != 6:
            raise ValueError(
                "Expected videos with shape (batch, views, channels, frames, height, width) "
                "or (batch, channels, frames, height, width), "
                f"got {tuple(x.shape)}"
            )
        batch, views = x.shape[:2]
        if views == 0:
            raise ValueError("videos must contain at least one view")

        # TimeSformer encodes each view as a separate video in the batch.
        out = self.video_encoder(x.reshape(batch * views, *x.shape[2:]))
        if out.ndim != 2 or out.shape != (batch * views, self.agp.input_dim):
            raise ValueError(
                f"Expected video_encoder output with shape ({batch * views}, "
                f"{self.agp.input_dim}), got {tuple(out.shape)}"
            )
        out = out.reshape(batch, views, self.agp.input_dim)
        out = self.agp(out)
        return out

class AttentiveGatedProjector(nn.Module):
    """Fuse view features and project them into the language embedding space.

    Inputs have shape ``(batch, views, input_dim)``, including an explicit
    view dimension for a single view. Outputs have shape
    ``(batch, out_dim)``. After attention and mean pooling, normalized hidden
    features ``h`` receive a gated residual: ``h + h * sigmoid(W h + b)``.
    """

    def __init__(
        self,
        input_dim=400,
        hidden_dim=1024,
        out_dim=576,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("out_dim", out_dim),
            ("num_heads", num_heads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.input_dim = input_dim
        self.view_norm = nn.LayerNorm(input_dim)
        self.view_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.vision_language_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, video_feats):
        """Attend across views, fuse them, and return one embedding per sample."""
        if video_feats.ndim != 3 or video_feats.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected video_feats with shape (batch, views, {self.input_dim}), "
                f"got {tuple(video_feats.shape)}"
            )
        if video_feats.shape[1] == 0:
            raise ValueError("video_feats must contain at least one view")

        views = self.view_norm(video_feats)
        attended, _ = self.view_attn(views, views, views, need_weights=False)
        fused = attended.mean(dim=1)
        hidden = self.norm1(self.proj1(fused))
        hidden = hidden + hidden * self.gate(hidden)
        hidden = self.norm2(self.proj2(hidden))
        return self.vision_language_proj(hidden)
