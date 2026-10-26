"""Small camera-supported velocity adapters for the frozen DiT4DiT policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def build_camera_token_layout(
    *,
    hidden: torch.Tensor,
    pixel_height: int,
    pixel_width: int,
    camera_split: float,
    patch_size: tuple[int, int, int],
    vae_spatial_scale: int,
    vae_temporal_scale: int,
    num_frames_in: int,
    num_frames_out: int,
) -> dict[str, Any]:
    """Map observed scene/wrist tiles onto the installed Cosmos token order."""
    if hidden.ndim != 3:
        raise ValueError(f"Expected flattened [B,S,D] hidden, got {tuple(hidden.shape)}")
    if not 0.0 < camera_split < 1.0:
        raise ValueError("camera_split must be inside (0,1)")
    if len(patch_size) != 3 or patch_size[0] != 1:
        raise ValueError(f"This route requires temporal patch size 1, got {patch_size}")
    if pixel_width != 2 * pixel_height or camera_split != 0.5:
        raise ValueError(
            f"Expected equal horizontal scene/wrist tiles with 2:1 aspect, got {pixel_height}x{pixel_width}"
        )

    p_t, p_h, p_w = (int(v) for v in patch_size)
    spatial_scale = int(vae_spatial_scale)
    temporal_scale = max(1, int(vae_temporal_scale))
    latent_t = (int(num_frames_out) - 1) // temporal_scale + 1
    latent_h = int(pixel_height) // spatial_scale
    latent_w = int(pixel_width) // spatial_scale
    grid_t, grid_h, grid_w = latent_t // p_t, latent_h // p_h, latent_w // p_w
    if min(grid_t, grid_h, grid_w) <= 0 or grid_t * grid_h * grid_w != hidden.shape[1]:
        raise ValueError(
            "Cosmos token-grid mismatch: "
            f"hidden S={hidden.shape[1]}, derived grid={(grid_t, grid_h, grid_w)} "
            f"from pixels={(pixel_height, pixel_width)}, patch={patch_size}, "
            f"num_frames_out={num_frames_out}"
        )

    observed_t = min(grid_t, (int(num_frames_in) - 1) // temporal_scale + 1)
    if observed_t < 1:
        raise ValueError("No observation-conditioned latent slot was found")

    # Patch columns are ordered left-to-right. Exclude any patch crossing the tile seam.
    patch_width_px = spatial_scale * p_w
    seam_px = int(round(pixel_width * camera_split))
    scene_cols = [c for c in range(grid_w) if (c + 1) * patch_width_px <= seam_px]
    wrist_cols = [c for c in range(grid_w) if c * patch_width_px >= seam_px]
    if not scene_cols or not wrist_cols:
        raise ValueError("The camera seam leaves an empty readout support")

    scene_mask = torch.zeros(hidden.shape[1], device=hidden.device, dtype=torch.bool)
    wrist_mask = torch.zeros_like(scene_mask)
    for t in range(observed_t):
        for row in range(grid_h):
            start = (t * grid_h + row) * grid_w
            scene_mask[start + torch.tensor(scene_cols, device=hidden.device)] = True
            wrist_mask[start + torch.tensor(wrist_cols, device=hidden.device)] = True

    return {
        "scene_mask": scene_mask,
        "wrist_mask": wrist_mask,
        "grid_shape": (grid_t, grid_h, grid_w),
        "observed_time_slots": observed_t,
        "flatten_order": "time_row_column",
        "camera_split_px": seam_px,
        "patch_width_px": patch_width_px,
    }


def _pool_supported(hidden: torch.Tensor, token_layout: dict[str, Any], key: str) -> torch.Tensor:
    mask = token_layout[key]
    if mask.ndim != 1 or mask.shape[0] != hidden.shape[1] or not bool(mask.any()):
        raise ValueError(f"Invalid {key} for hidden shape {tuple(hidden.shape)}")
    return hidden[:, mask.to(device=hidden.device)].mean(dim=1)


class _CameraVelocityExpert(nn.Module):
    def __init__(self, action_hidden_dim: int, video_hidden_dim: int, state_dim: int, action_dim: int, rank: int):
        super().__init__()
        self.query = nn.Linear(action_hidden_dim, rank, bias=False)
        self.key = nn.Linear(video_hidden_dim, rank, bias=False)
        self.value = nn.Linear(video_hidden_dim, rank, bias=False)
        self.state = nn.Linear(state_dim, rank, bias=False)
        self.time = nn.Linear(17, rank, bias=False)
        self.norm = nn.LayerNorm(4 * rank)
        self.mlp = nn.Sequential(
            nn.Linear(4 * rank, 2 * rank),
            nn.GELU(),
            nn.Linear(2 * rank, rank),
            nn.GELU(),
            nn.Linear(rank, action_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _time_features(tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1).float()
        freqs = torch.pow(2.0, torch.arange(8, device=tau.device, dtype=tau.dtype)) * torch.pi
        phase = tau * freqs[None, :]
        return torch.cat([tau, torch.sin(phase), torch.cos(phase)], dim=-1)

    def forward(
        self,
        action_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        token_layout: dict[str, Any],
        tau: torch.Tensor,
        state: torch.Tensor | None,
        support: str,
    ) -> torch.Tensor:
        x = action_hidden.float()
        camera_hidden = video_hidden.float()[:, token_layout[support].to(video_hidden.device), :]
        if camera_hidden.shape[1] == 0:
            raise ValueError(f"Empty readout support {support}")
        q = self.query(x)
        k = self.key(camera_hidden)
        v = self.value(camera_hidden)
        weights = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=-1)
        context = torch.matmul(weights, v)

        if state is None:
            state_vec = torch.zeros(x.shape[0], self.state.in_features, device=x.device, dtype=x.dtype)
        else:
            state_vec = state.float().reshape(state.shape[0], -1)
            if state_vec.shape[-1] != self.state.in_features:
                raise ValueError(
                    f"Expected adapter state dim {self.state.in_features}, got {state_vec.shape[-1]}"
                )
        time_vec = self._time_features(tau.to(x.device))
        state_features = self.state(state_vec).unsqueeze(1).expand(-1, x.shape[1], -1)
        time_features = self.time(time_vec).unsqueeze(1).expand(-1, x.shape[1], -1)
        features = torch.cat([q, context, state_features, time_features], dim=-1)
        return self.mlp(self.norm(features)).to(dtype=action_hidden.dtype)


class CameraSupportedVelocityResidualAdapter(nn.Module):
    """Two low-rank residual experts plus the current-observation three-way router."""

    ROUTE_TO_INDEX = {"0": 0, "S": 1, "W": 2}

    def __init__(
        self,
        action_hidden_dim: int,
        video_hidden_dim: int,
        state_dim: int,
        action_dim: int,
        rank: int = 64,
        router_hidden: int = 128,
    ):
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                "S": _CameraVelocityExpert(action_hidden_dim, video_hidden_dim, state_dim, action_dim, rank),
                "W": _CameraVelocityExpert(action_hidden_dim, video_hidden_dim, state_dim, action_dim, rank),
            }
        )
        self.router = nn.Sequential(
            nn.LayerNorm(2 * video_hidden_dim + state_dim),
            nn.Linear(2 * video_hidden_dim + state_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 3),
        )
        self.video_hidden_dim = video_hidden_dim
        self.state_dim = state_dim

    def route_logits(
        self, video_hidden: torch.Tensor, token_layout: dict[str, Any], state: torch.Tensor | None
    ) -> torch.Tensor:
        scene = _pool_supported(video_hidden, token_layout, "scene_mask").float()
        wrist = _pool_supported(video_hidden, token_layout, "wrist_mask").float()
        if state is None:
            state_vec = torch.zeros(video_hidden.shape[0], self.state_dim, device=video_hidden.device)
        else:
            state_vec = state.float().reshape(state.shape[0], -1)
            if state_vec.shape[-1] != self.state_dim:
                raise ValueError(f"Expected router state dim {self.state_dim}, got {state_vec.shape[-1]}")
        return self.router(torch.cat([scene, wrist, state_vec], dim=-1))

    def residual(
        self,
        route: str,
        action_hidden: torch.Tensor,
        video_hidden: torch.Tensor,
        token_layout: dict[str, Any],
        tau: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        if route not in self.experts:
            raise ValueError(f"Residual route must be S or W, got {route!r}")
        support = "scene_mask" if route == "S" else "wrist_mask"
        return self.experts[route](action_hidden, video_hidden, token_layout, tau, state, support)
