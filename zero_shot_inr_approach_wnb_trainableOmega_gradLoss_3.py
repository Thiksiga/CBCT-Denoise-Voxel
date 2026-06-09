# -*- coding: utf-8 -*-
"""
Zero-Shot Implicit Neural Representation (INR) Pipeline
========================================================

Single-file research pipeline for patch-wise SIREN fitting on 3D CBCT volumes,
overlap-add reconstruction, and quantitative evaluation (SSIM, LPIPS, attributions).

Sections
--------
1. Imports and runtime configuration
2. Volume patching (MONAI)
3. Differential operators (autograd)
4. Coordinate grids and image preprocessing
4b. Noise model v2 (overlapping Poisson, per-patch fixed noise)
5. SIREN architecture
6. PyTorch datasets
7. Visualization utilities (W&B; TensorBoard disabled)
8. LPIPS metric helpers
9. Training and evaluation (script entry)
"""

# -----------------------------------------------------------------------------
# 1. Imports and runtime configuration
# -----------------------------------------------------------------------------

import os
import random
import time

import captum
import psutil
import lpips
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import wandb
from captum.attr import IntegratedGradients, LayerActivation
from monai.data.utils import dense_patch_slices
from monai.inferers import sliding_window_inference
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.special import factorial
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

_NVML_INITIALIZED = False

# W&B account: thiksigar (thiksigar-university-of-peradeniya)
WANDB_API_KEY = (
    "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"
)
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY, relogin=True)

_DEFAULT_INPUT = r"/projects1/Toothfairy/ToothFairy_Dataset/Dataset/Dataset/P10/data.npy"

wandb.init(
    project="zero-shot-inr",
    name=f"SIREN_zero_shot_2_MSE_gradLoss_20steps_halfPrecision{os.path.splitext(os.path.basename(_DEFAULT_INPUT))[0]}",
    config={
        # Model
        "architecture": "SIREN",
        "training_mode": "zero_shot",
        "comit": "steps20 mse_grad_tv_noiseJS half precision",
        "hidden_features": 256,
        "hidden_layers": 3,
        "outermost_linear": True,
        # SIREN frequency (trainable nn.Parameter per SineLayer; same role as LR in config)
        "first_omega_0": 60,
        "hidden_omega_0": 60,
        # Data
        "input_data_name": _DEFAULT_INPUT,
        # 64³ / stride 32³ = fewer patches & less disk than 32³ / stride 16³ (~8× fewer per axis)
        "patch_size": [32, 32, 32],
        "stride": [16, 16, 16],
        # num_patches is set after MONAI extraction (volume shape + patch_size + stride)
        # Optimisation
        "learning_rate": 1e-5,
        "optimizer": "Adam",  # Adam | AdamW | SGD
        "loss_function": "mse + NoiseLoss", #"mse + gradLoss + tv + residualNoiseJS",
        "denoising_strategy": "INR clean estimate", #"INR clean estimate + TV + residual noise distribution matching",
        "synthetic_noise_target": "residual",
        "synthetic_noise_direct_output_loss_enabled": False,
        "noise_reference_scope": "global_volume",
        "noise_patch_reference_mode": "slice_from_global_noise_field",
        "noise_normalization": "fixed_hu_offset_(hu+1001)*100/6264",
        "noise_rescale_transform": "log_rescale_original_noise_modeling",
        "activation_snapshot_log_js": True,
        "loss_mse_weight": 1.0,
        "loss_grad_weight": 1.0,
        # Keep direct output-vs-synthetic-noise terms disabled for denoising.
        # Synthetic noise should describe the residual removed from the patch,
        # not the generated clean estimate.
        "loss_noise_mse_weight": 0.0,
        "loss_noise_js_weight": 0,
        "loss_tv_weight": 1e-5,
        "loss_residual_noise_js_weight": 0.001, #0.001,
        "noise_model_a": 1.0,
        "noise_model_b": 0.01,
        "noise_model_hu_bin_width": 200,
        "noise_model_overlap_ratio": 0.05,
        "noise_model_bins": 100,
        "noise_model_seed": 42,
        "total_steps": 20,
        "steps_til_summary": 10,
        # Logging — hardware
        "log_hardware_every": 1,
        "gpu_device_index": 0,
        # Logging — activation maps (subset of neurons saved as PNG)
        "activation_snapshot_num_patches": 50,
        "activation_num_neurons_to_log": 50,
        "activation_log_seed": 42,
        "activation_slice_axis": 2,
        "activation_slice_idx": None,  # None = center slice; auto-clamped to patch size
        "activation_maps_dir": "activation_maps",
        "activation_log_wandb_images": False,  # 256 PNGs/patch fills W&B staging disk
        "activation_log_montage": True,  # one overview image per snapshot to W&B
        "activation_wandb_batch_size": 32,
        "activation_log_artifact": False,
        # Trainable omega tracking (one scalar omega_0 per SineLayer)
        "omega_track_seed": 42,
        "omega_plot_path": "omega_tracked.png",
        # System resource plots (logged to W&B at end of training)
        "system_plot_path": "system_resources.png",
    },
)

cfg = wandb.config
patch_size = tuple(cfg.patch_size)
stride = tuple(cfg.stride)


def build_optimizer(model, optimizer_name: str, learning_rate: float):
    """Instantiate optimiser from W&B config."""
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _init_nvml():
    global _NVML_INITIALIZED
    if _NVML_AVAILABLE and not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True


def get_hardware_metrics(device_index=0):
    """Collect host RAM, CPU, and GPU memory / utilisation for W&B."""
    metrics = {}

    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    metrics["system/ram_used_gb"] = vm.used / (1024 ** 3)
    metrics["system/ram_available_gb"] = vm.available / (1024 ** 3)
    metrics["system/ram_total_gb"] = vm.total / (1024 ** 3)
    metrics["system/ram_percent"] = vm.percent
    metrics["system/process_rss_gb"] = proc.memory_info().rss / (1024 ** 3)
    metrics["system/cpu_utilization_percent"] = psutil.cpu_percent(interval=None)

    if torch.cuda.is_available():
        metrics["system/compute_device"] = f"cuda:{device_index}"
        metrics["system/gpu_name"] = torch.cuda.get_device_name(device_index)
        metrics["system/gpu_memory_allocated_gb"] = (
            torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        )
        metrics["system/gpu_memory_reserved_gb"] = (
            torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        )
        metrics["system/gpu_max_memory_allocated_gb"] = (
            torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
        )

        if _NVML_AVAILABLE:
            _init_nvml()
            gpu_count = pynvml.nvmlDeviceGetCount()
            metrics["system/gpu_count"] = gpu_count
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                prefix = f"system/gpu_{i}"
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")
                metrics[f"{prefix}/name"] = gpu_name
                metrics[f"{prefix}/utilization_percent"] = float(util.gpu)
                metrics[f"{prefix}/memory_utilization_percent"] = float(util.memory)
                metrics[f"{prefix}/memory_used_gb"] = mem.used / (1024 ** 3)
                metrics[f"{prefix}/memory_total_gb"] = mem.total / (1024 ** 3)

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["system/gpu_utilization_percent"] = float(util.gpu)
            metrics["system/gpu_memory_utilization_percent"] = float(util.memory)
            metrics["system/gpu_memory_used_gb"] = mem.used / (1024 ** 3)
            metrics["system/gpu_memory_total_gb"] = mem.total / (1024 ** 3)
    else:
        metrics["system/compute_device"] = "cpu"

    return metrics


# Keys recorded each step for end-of-run resource plots
SYSTEM_HISTORY_KEYS = (
    "system/ram_used_gb",
    "system/ram_percent",
    "system/process_rss_gb",
    "system/gpu_utilization_percent",
    "system/gpu_memory_used_gb",
    "system/gpu_memory_allocated_gb",
    "system/gpu_memory_utilization_percent",
)


def new_system_history():
    """Empty time-series buffers for RAM / GPU plots."""
    history = {key: [] for key in SYSTEM_HISTORY_KEYS}
    history["_steps"] = []
    return history


def append_system_history(history, metrics, global_step):
    """Append one hardware sample per tracked metric (NaN if unavailable)."""
    history["_steps"].append(global_step)
    for key in SYSTEM_HISTORY_KEYS:
        history[key].append(metrics.get(key, float("nan")))


def sample_activation_snapshots(num_patches, total_steps, num_snapshot_patches, seed):
    """Pick random patches and one random training step per patch."""
    rng = random.Random(seed)
    n = min(int(num_snapshot_patches), num_patches)
    patch_indices = sorted(rng.sample(range(num_patches), n))
    step_by_patch = {pi: rng.randrange(total_steps) for pi in patch_indices}
    return set(patch_indices), step_by_patch


def sample_activation_neuron_ids(total_neurons, num_neurons_to_log, seed):
    """Pick a fixed random subset of neuron indices to save as activation maps."""
    rng = random.Random(seed)
    n = min(int(num_neurons_to_log), int(total_neurons))
    return sorted(rng.sample(range(int(total_neurons)), n))


# -----------------------------------------------------------------------------
# 2. Volume patching (MONAI)
# -----------------------------------------------------------------------------

def extract_patches_monai(volume: torch.Tensor, patch_size, stride):
    """
    Extract overlapping 3D patches from a volume using MONAI slice generation.

    Args:
        volume: torch.Tensor of shape [H, W, D]
        patch_size: tuple (ph, pw, pd)
        stride: tuple (sh, sw, sd)

    Returns:
        patches: list of dicts with keys:
            - "patch": tensor [ph, pw, pd]
            - "slice": tuple of slice objects
        image_shape: original spatial shape
    """
    if volume.dim() != 3:
        raise ValueError("volume must have shape [H, W, D]")

    image_shape = tuple(volume.shape)
    patch_slices = dense_patch_slices(
        image_size=image_shape,
        patch_size=patch_size,
        scan_interval=stride,
    )

    patches = []
    for s in patch_slices:
        patches.append({
            "patch": volume[s].clone(),
            "slice": s,
        })

    return patches, image_shape


def reconstruct_patches_monai(patches, image_shape):
    """
    Reconstruct a 3D volume from MONAI-generated patch dictionaries.

    Args:
        patches: list of dicts with:
            - "patch": tensor [ph, pw, pd]
            - "slice": tuple of slice objects
        image_shape: tuple (H, W, D)

    Returns:
        recon: reconstructed tensor [H, W, D]
        count: overlap count tensor [H, W, D]
    """
    print(f"reconstruct_patches_monai: {len(patches)}")
    if len(patches) == 0:
        raise ValueError("patches list is empty")

    dtype = patches[0]["patch"].dtype
    device = patches[0]["patch"].device

    recon = torch.zeros(image_shape, dtype=dtype, device=device)
    count = torch.zeros(image_shape, dtype=dtype, device=device)

    for item in patches:
        patch = item["patch"]
        s = item["slice"]

        recon[s] += patch
        count[s] += 1

    recon = recon / count.clamp_min(1)
    return recon, count


# -----------------------------------------------------------------------------
# 3. Differential operators (autograd)
# -----------------------------------------------------------------------------

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


# -----------------------------------------------------------------------------
# 4. Coordinate grids and image preprocessing
# -----------------------------------------------------------------------------

def img_normalisation(img: torch.Tensor):
    """
    Converts image to range [-1, 1]
    """

    img = img.float()

    # Known CT range
    min_val = -1000
    max_val = 5264

    # shift + scale → [0,1]
    img = (img - min_val) / (max_val - min_val)

    # scale → [-1,1]
    img = img * 2 - 1

    return img


# -----------------------------------------------------------------------------
# 4b. Noise model v2 (from noise_modeling2.py — overlapping Poisson + uniform)
# -----------------------------------------------------------------------------

def _tensor_to_hu(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1.0) * 0.5 * 6264.0 - 1000.0


def _normalize_patch_offset(hu_patch: np.ndarray) -> np.ndarray:
    """Fixed HU offset normalisation from noise_modeling.py."""
    return (hu_patch + 1001.0) * 100.0 / 6264.0


def _build_overlapping_range_poisson_distribution(
    img_offset,
    hu_bin_width=200,
    overlap_ratio=0.05,
    original_range=6264.0,
    normalized_range=100.0,
):
    bin_width = hu_bin_width * normalized_range / original_range
    l_b = float(np.min(img_offset))
    u_b = float(np.max(img_offset))
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError(f"Invalid Poisson bin width: {bin_width}")
    if overlap_ratio < 0 or overlap_ratio >= 1:
        raise ValueError("overlap_ratio must be in [0, 1).")
    if (u_b - l_b) < 1e-8:
        return {l_b: 1.0}, l_b, u_b

    overlap = bin_width * overlap_ratio
    step = bin_width - overlap

    merged_distribution = {}
    start = l_b
    while start < u_b:
        end = start + bin_width
        region_mask = (img_offset >= start) & (img_offset <= end)
        region_pixels = img_offset[region_mask]
        if region_pixels.size == 0:
            start += step
            continue
        region_mean = float(np.mean(region_pixels))
        for bi in region_pixels:
            probability_bi = (region_mean ** bi) * np.exp(-region_mean) / factorial(bi)
            if np.isfinite(probability_bi) and probability_bi > 0:
                merged_distribution[bi] = merged_distribution.get(bi, 0.0) + probability_bi
        start += step

    if not merged_distribution:
        flat = img_offset.reshape(-1).astype(np.float32)
        unique_values, counts = np.unique(flat, return_counts=True)
        probabilities = counts.astype(np.float32) / counts.sum()
        return dict(zip(unique_values.tolist(), probabilities.tolist())), l_b, u_b

    distribution_sorted = {k: merged_distribution[k] for k in sorted(merged_distribution)}
    total_probability = sum(distribution_sorted.values())
    if not np.isfinite(total_probability) or total_probability <= 0:
        flat = img_offset.reshape(-1).astype(np.float32)
        unique_values, counts = np.unique(flat, return_counts=True)
        probabilities = counts.astype(np.float32) / counts.sum()
        return dict(zip(unique_values.tolist(), probabilities.tolist())), l_b, u_b

    distribution_sorted = {k: v / total_probability for k, v in distribution_sorted.items()}
    return distribution_sorted, l_b, u_b


def _generate_custom_poisson_noise(distribution_sorted, shape, seed=42):
    rng = np.random.default_rng(seed)
    keys = np.array(list(distribution_sorted.keys()), dtype=np.float32)
    values = np.array(list(distribution_sorted.values()), dtype=np.float32)
    values_normalized = values / values.sum()
    noise_flat = rng.choice(keys, size=np.prod(shape), p=values_normalized)
    return noise_flat.reshape(shape)


def _generate_random_noise(shape, l_b, u_b, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=l_b, high=u_b, size=shape)


def _transform_and_rescale_noise(combined_noise, u_b, l_b):
    combined_noise = np.asarray(combined_noise, dtype=np.float32)
    max_val = np.max(combined_noise)
    combined_log = u_b - np.log(combined_noise / max_val + 1e-12)
    rescaled = u_b - combined_log
    rescaled_offset = rescaled - np.min(rescaled)
    denominator = np.max(rescaled_offset) - np.min(rescaled_offset)
    if denominator < 1e-12:
        return np.full_like(rescaled_offset, l_b)
    return rescaled_offset * ((u_b - l_b) / denominator) + l_b


def build_global_noise_reference(
    normalized_volume: torch.Tensor,
    a=1.0,
    b=0.01,
    hu_bin_width=200,
    overlap_ratio=0.05,
    bins=100,
    seed=42,
):
    """
    Estimate one synthetic noise field from the full normalized volume.

    Patch-level noise references should then be spatial slices from this field,
    so adjacent patches share one consistent noise coordinate system.
    """
    hu_volume = _tensor_to_hu(normalized_volume.detach()).cpu().numpy()
    original_range = float(hu_volume.max() - hu_volume.min())
    if original_range < 1e-8:
        original_range = 6264.0

    img_offset = _normalize_patch_offset(hu_volume)
    shifted = hu_volume + 1001.0
    shift_min = float(np.min(shifted))
    shift_max = float(np.max(shifted))

    distribution_sorted, l_b, u_b = _build_overlapping_range_poisson_distribution(
        img_offset,
        hu_bin_width=hu_bin_width,
        overlap_ratio=overlap_ratio,
        original_range=original_range,
    )

    shape = tuple(normalized_volume.shape)
    poisson_noise = _generate_custom_poisson_noise(distribution_sorted, shape, seed=seed)
    uniform_noise = _generate_random_noise(shape, l_b, u_b, seed=seed + 1)
    combined_noise = a * poisson_noise + b * uniform_noise
    rescaled_noise = _transform_and_rescale_noise(combined_noise, u_b, l_b)

    ref_flat = rescaled_noise.reshape(-1)
    common_min = float(ref_flat.min())
    common_max = float(ref_flat.max())
    if common_max - common_min < 1e-8:
        common_max = common_min + 1.0

    pdf_np, edges_np = _compute_pdf_np(ref_flat, bins=bins, data_range=(common_min, common_max))
    centered_ref = ref_flat - float(ref_flat.mean())
    centered_min = float(centered_ref.min())
    centered_max = float(centered_ref.max())
    if centered_max - centered_min < 1e-8:
        centered_max = centered_min + 1.0
    residual_pdf_np, residual_edges_np = _compute_pdf_np(
        centered_ref, bins=bins, data_range=(centered_min, centered_max)
    )

    return {
        "noise_offset": torch.tensor(rescaled_noise, dtype=torch.float32),
        "shift_min": shift_min,
        "shift_max": shift_max,
        "offset_min": float(l_b),
        "offset_max": float(u_b),
        "hu_min": float(hu_volume.min()),
        "hu_max": float(hu_volume.max()),
        "hu_range": float(hu_volume.max() - hu_volume.min()),
        "reference_noise_mean": float(ref_flat.mean()),
        "reference_noise_std": float(ref_flat.std()),
        "reference_residual_std": float(centered_ref.std()),
        "reference_pdf": torch.tensor(pdf_np, dtype=torch.float32),
        "bin_edges": torch.tensor(edges_np, dtype=torch.float32),
        "reference_residual_pdf": torch.tensor(residual_pdf_np, dtype=torch.float32),
        "residual_bin_edges": torch.tensor(residual_edges_np, dtype=torch.float32),
    }


def _compute_pdf_np(data, bins=100, data_range=None, eps=1e-12):
    hist, bin_edges = np.histogram(data, bins=bins, range=data_range, density=False)
    pdf = hist.astype(np.float32)
    pdf /= np.sum(pdf) + eps
    pdf = np.clip(pdf, eps, None)
    pdf /= np.sum(pdf)
    return pdf, bin_edges


def _js_divergence_np(P, Q, eps=1e-12):
    P = np.clip(np.asarray(P, dtype=np.float32), eps, None)
    Q = np.clip(np.asarray(Q, dtype=np.float32), eps, None)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    return float(0.5 * np.sum(P * np.log(P / M)) + 0.5 * np.sum(Q * np.log(Q / M)))


def _soft_histogram_torch(values, bin_edges, sigma=None, eps=1e-12):
    values = values.reshape(-1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    if sigma is None:
        sigma = width * 0.5
    diff = values.unsqueeze(1) - centers.unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    hist = weights.sum(dim=0) + eps
    return hist / hist.sum()


def _js_divergence_torch(P, Q, eps=1e-12):
    P = torch.clamp(P, min=eps)
    Q = torch.clamp(Q, min=eps)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    return 0.5 * torch.sum(P * torch.log(P / M)) + 0.5 * torch.sum(Q * torch.log(Q / M))


def _tensor_to_patch_offset(tensor, shift_min, shift_max):
    """Map [-1,1] INR tensor to fixed noise-model offset space."""
    hu = _tensor_to_hu(tensor)
    return (hu + 1001.0) * 100.0 / 6264.0


def tv_loss_3d(volume: torch.Tensor) -> torch.Tensor:
    """3D total-variation loss to discourage high-frequency noisy texture."""
    dz = torch.abs(volume[1:, :, :] - volume[:-1, :, :]).mean()
    dy = torch.abs(volume[:, 1:, :] - volume[:, :-1, :]).mean()
    dx = torch.abs(volume[:, :, 1:] - volume[:, :, :-1]).mean()
    return dx + dy + dz


class PatchNoiseModel:
    """
    Fit overlapping Poisson on a patch, generate synthetic noise once,
    and reuse the same noise field for all training steps on that patch.
    """

    def __init__(
        self,
        a=1.0,
        b=0.01,
        hu_bin_width=200,
        overlap_ratio=0.05,
        bins=100,
    ):
        self.a = float(a)
        self.b = float(b)
        self.hu_bin_width = int(hu_bin_width)
        self.overlap_ratio = float(overlap_ratio)
        self.bins = int(bins)
        self.reference_noise_offset = None
        self.shift_min = None
        self.shift_max = None
        self.reference_pdf = None
        self.bin_edges = None
        self.reference_residual_pdf = None
        self.residual_bin_edges = None
        self.patch_js_at_setup = None
        self.setup_metrics = {}

    def setup_patch(self, ground_truth, patch_shape, seed, device):
        hu_patch = _tensor_to_hu(ground_truth.detach()).cpu().numpy().reshape(patch_shape)
        original_range = float(hu_patch.max() - hu_patch.min())
        if original_range < 1e-8:
            original_range = 6264.0

        img_offset = _normalize_patch_offset(hu_patch)
        shifted = hu_patch + 1001.0
        self.shift_min = float(np.min(shifted))
        self.shift_max = float(np.max(shifted))

        distribution_sorted, l_b, u_b = _build_overlapping_range_poisson_distribution(
            img_offset,
            hu_bin_width=self.hu_bin_width,
            overlap_ratio=self.overlap_ratio,
            original_range=original_range,
        )

        shape = tuple(patch_shape)
        poisson_noise = _generate_custom_poisson_noise(distribution_sorted, shape, seed=seed)
        uniform_noise = _generate_random_noise(shape, l_b, u_b, seed=seed + 1)
        combined_noise = self.a * poisson_noise + self.b * uniform_noise
        rescaled_noise = _transform_and_rescale_noise(combined_noise, u_b, l_b)

        self.reference_noise_offset = torch.tensor(
            rescaled_noise, dtype=torch.float32, device=device
        )

        ref_flat = rescaled_noise.reshape(-1)
        common_min = float(ref_flat.min())
        common_max = float(ref_flat.max())
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0

        pdf_np, edges_np = _compute_pdf_np(
            ref_flat, bins=self.bins, data_range=(common_min, common_max)
        )
        self.reference_pdf = torch.tensor(pdf_np, dtype=torch.float32, device=device)
        self.bin_edges = torch.tensor(edges_np, dtype=torch.float32, device=device)

        centered_ref = ref_flat - float(ref_flat.mean())
        centered_min = float(centered_ref.min())
        centered_max = float(centered_ref.max())
        if centered_max - centered_min < 1e-8:
            centered_max = centered_min + 1.0
        residual_pdf_np, residual_edges_np = _compute_pdf_np(
            centered_ref, bins=self.bins, data_range=(centered_min, centered_max)
        )
        self.reference_residual_pdf = torch.tensor(
            residual_pdf_np, dtype=torch.float32, device=device
        )
        self.residual_bin_edges = torch.tensor(
            residual_edges_np, dtype=torch.float32, device=device
        )
        self.patch_js_at_setup = _js_divergence_np(pdf_np, pdf_np)
        self.setup_metrics = {
            "noise/setup_hu_min": float(hu_patch.min()),
            "noise/setup_hu_max": float(hu_patch.max()),
            "noise/setup_hu_range": float(hu_patch.max() - hu_patch.min()),
            "noise/setup_offset_min": float(l_b),
            "noise/setup_offset_max": float(u_b),
            "noise/setup_reference_noise_mean": float(ref_flat.mean()),
            "noise/setup_reference_noise_std": float(ref_flat.std()),
            "noise/setup_reference_noise_min": float(ref_flat.min()),
            "noise/setup_reference_noise_max": float(ref_flat.max()),
            "noise/setup_reference_residual_std": float(centered_ref.std()),
        }

        return self

    def setup_from_global(self, ground_truth, patch_shape, patch_slice, global_reference, device):
        """
        Use a spatial crop from the full-volume synthetic noise field.

        This avoids estimating unrelated noise distributions for neighboring
        patches, which can otherwise produce patch-wise intensity/noise bands.
        """
        hu_patch = _tensor_to_hu(ground_truth.detach()).cpu().numpy().reshape(patch_shape)
        self.shift_min = float(global_reference["shift_min"])
        self.shift_max = float(global_reference["shift_max"])

        rescaled_noise = (
            global_reference["noise_offset"][patch_slice]
            .detach()
            .cpu()
            .numpy()
            .reshape(patch_shape)
        )
        self.reference_noise_offset = torch.tensor(
            rescaled_noise, dtype=torch.float32, device=device
        )

        ref_flat = rescaled_noise.reshape(-1)
        common_min = float(ref_flat.min())
        common_max = float(ref_flat.max())
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0

        pdf_np, edges_np = _compute_pdf_np(
            ref_flat, bins=self.bins, data_range=(common_min, common_max)
        )
        self.reference_pdf = torch.tensor(pdf_np, dtype=torch.float32, device=device)
        self.bin_edges = torch.tensor(edges_np, dtype=torch.float32, device=device)

        centered_ref = ref_flat - float(ref_flat.mean())
        centered_min = float(centered_ref.min())
        centered_max = float(centered_ref.max())
        if centered_max - centered_min < 1e-8:
            centered_max = centered_min + 1.0
        residual_pdf_np, residual_edges_np = _compute_pdf_np(
            centered_ref, bins=self.bins, data_range=(centered_min, centered_max)
        )
        self.reference_residual_pdf = torch.tensor(
            residual_pdf_np, dtype=torch.float32, device=device
        )
        self.residual_bin_edges = torch.tensor(
            residual_edges_np, dtype=torch.float32, device=device
        )
        self.patch_js_at_setup = _js_divergence_np(pdf_np, pdf_np)
        self.setup_metrics = {
            "noise/setup_hu_min": float(hu_patch.min()),
            "noise/setup_hu_max": float(hu_patch.max()),
            "noise/setup_hu_range": float(hu_patch.max() - hu_patch.min()),
            "noise/setup_offset_min": float(global_reference["offset_min"]),
            "noise/setup_offset_max": float(global_reference["offset_max"]),
            "noise/setup_reference_noise_mean": float(ref_flat.mean()),
            "noise/setup_reference_noise_std": float(ref_flat.std()),
            "noise/setup_reference_noise_min": float(ref_flat.min()),
            "noise/setup_reference_noise_max": float(ref_flat.max()),
            "noise/setup_reference_residual_std": float(centered_ref.std()),
            "noise/setup_source": "global_volume_slice",
        }

        return self

    def noise_mse_loss(self, model_output):
        """MSE between model output (offset space) and fixed synthetic patch noise."""
        model_offset = _tensor_to_patch_offset(
            model_output, self.shift_min, self.shift_max
        ).reshape(-1)
        target = self.reference_noise_offset.reshape(-1)
        return F.mse_loss(model_offset, target)

    def noise_js_loss(self, model_output):
        """JS(PDF(model output offset), PDF(fixed synthetic patch noise))."""
        model_offset = _tensor_to_patch_offset(
            model_output, self.shift_min, self.shift_max
        ).reshape(-1)
        P = _soft_histogram_torch(model_offset, self.bin_edges)
        Q = self.reference_pdf / self.reference_pdf.sum()
        return _js_divergence_torch(P, Q)

    def residual_noise_js_loss(self, model_output, ground_truth):
        """JS(PDF(noisy input - clean estimate), PDF(centered synthetic noise))."""
        model_offset = _tensor_to_patch_offset(
            model_output, self.shift_min, self.shift_max
        )
        gt_offset = _tensor_to_patch_offset(
            ground_truth, self.shift_min, self.shift_max
        )
        residual_offset = (gt_offset - model_offset).reshape(-1)
        residual_offset = residual_offset - residual_offset.mean()
        P = _soft_histogram_torch(residual_offset, self.residual_bin_edges)
        Q = self.reference_residual_pdf / self.reference_residual_pdf.sum()
        return _js_divergence_torch(P, Q)

    def patch_js_numpy(self, model_output):
        """NumPy JS for logging (non-differentiable snapshot)."""
        model_offset = _tensor_to_patch_offset(
            model_output.detach(), self.shift_min, self.shift_max
        ).cpu().numpy().reshape(-1)
        ref_flat = self.reference_noise_offset.detach().cpu().numpy().reshape(-1)
        common_min = float(min(model_offset.min(), ref_flat.min()))
        common_max = float(max(model_offset.max(), ref_flat.max()))
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0
        P, _ = _compute_pdf_np(model_offset, bins=self.bins, data_range=(common_min, common_max))
        Q, _ = _compute_pdf_np(ref_flat, bins=self.bins, data_range=(common_min, common_max))
        return _js_divergence_np(P, Q)

    def residual_patch_js_numpy(self, model_output, ground_truth):
        """NumPy JS for logging residual-vs-synthetic-noise distribution."""
        model_offset = _tensor_to_patch_offset(
            model_output.detach(), self.shift_min, self.shift_max
        ).cpu().numpy().reshape(-1)
        gt_offset = _tensor_to_patch_offset(
            ground_truth.detach(), self.shift_min, self.shift_max
        ).cpu().numpy().reshape(-1)
        residual = gt_offset - model_offset
        residual = residual - float(residual.mean())

        ref_flat = self.reference_noise_offset.detach().cpu().numpy().reshape(-1)
        ref_flat = ref_flat - float(ref_flat.mean())

        common_min = float(min(residual.min(), ref_flat.min()))
        common_max = float(max(residual.max(), ref_flat.max()))
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0
        P, _ = _compute_pdf_np(residual, bins=self.bins, data_range=(common_min, common_max))
        Q, _ = _compute_pdf_np(ref_flat, bins=self.bins, data_range=(common_min, common_max))
        return _js_divergence_np(P, Q)


def create_3d_grid(H, D, W, device='cpu'):
    z = torch.linspace(-1, 1, steps=D, device=device)
    y = torch.linspace(-1, 1, steps=H, device=device)
    x = torch.linspace(-1, 1, steps=W, device=device)

    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    grid = torch.stack([zz, yy, xx], dim=-1)  # (D, H, W, 3)
    return grid


def create_3d_coords(H, D, W, device='cpu'):
    grid = create_3d_grid(H, D, W, device)
    coords = grid.view(-1, 3)  # (D*H*W, 3)
    return coords


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    # img = np.load(r"/content/drive/MyDrive/Mphil/P10/P10-20250324T100047Z-001/P10/data.npy")
    # img = torch.from_numpy(img[:, 150, :])
    print(f"img 1: {img.size}")
    img = img.resize((sidelength,)*2)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    print(f"img 2: {img.shape}")
    return img

# -----------------------------------------------------------------------------
# 5. SIREN architecture
# -----------------------------------------------------------------------------

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=10):
        super().__init__()
        # self.omega_0 = omega_0
        # self.is_first = is_first

        # self.in_features = in_features
        # self.linear = nn.Linear(in_features, out_features, bias=bias)

        # self.init_weights()
        # TRAINABLE OMEGA
        self.omega_0 = nn.Parameter(
            torch.tensor(float(omega_0))
        )

        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias
        )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=10, hidden_omega_0=10.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def get_sine_layers(model: nn.Module):
    """Return all SineLayer modules in order (each has one trainable omega_0)."""
    return [m for m in model.modules() if isinstance(m, SineLayer)]


def pick_tracked_omega_layer(model: nn.Module, seed: int = 42):
    """Pick one random SineLayer to monitor omega_0 during training."""
    layers = get_sine_layers(model)
    if not layers:
        raise ValueError("No SineLayer found in model")
    layer_idx = random.Random(seed).randrange(len(layers))
    return layer_idx, layers[layer_idx]


def collect_omega_metrics(model: nn.Module, tracked_layer_idx: int):
    """Scalar omega_0 values for W&B (tracked layer + all layers)."""
    layers = get_sine_layers(model)
    metrics = {
        "omega/tracked_layer_idx": tracked_layer_idx,
        "omega/tracked_value": layers[tracked_layer_idx].omega_0.detach().item(),
        "omega/tracked_is_first_layer": layers[tracked_layer_idx].is_first,
    }
    for i, layer in enumerate(layers):
        metrics[f"omega/layer_{i}"] = layer.omega_0.detach().item()
    return metrics


def collect_initial_omega_values(model: nn.Module):
    """Initial omega_0 per SineLayer for W&B config."""
    layers = get_sine_layers(model)
    return {f"omega_initial/layer_{i}": layer.omega_0.detach().item() for i, layer in enumerate[SineLayer](layers)}


def plot_omega_tracking(history, tracked_layer_idx, is_first_layer, out_path):
    """
    Plot tracked omega_0 vs global training step and log PNG to W&B.

    history: list of (global_step, omega_value) recorded after each optim.step.
    """
    if not history:
        return

    steps, values = zip(*history)
    initial = values[0]
    final = values[-1]
    delta = final - initial

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(initial, color="gray", linestyle="--", alpha=0.6, label=f"initial={initial:.6f}")
    ax.set_xlabel("Global step")
    ax.set_ylabel("omega_0")
    ax.set_title(
        f"Trainable omega_0 | tracked SineLayer {tracked_layer_idx} "
        f"({'first' if is_first_layer else 'hidden'})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    wandb.log({
        "omega/tracked_plot": wandb.Image(
            out_path,
            caption=(
                f"layer={tracked_layer_idx} | initial={initial:.6f} "
                f"final={final:.6f} | delta={delta:.6f}"
            ),
        ),
        "omega/tracked_initial": initial,
        "omega/tracked_final": final,
        "omega/tracked_delta": delta,
    })
    print(
        f"[omega] layer {tracked_layer_idx}: {initial:.6f} -> {final:.6f} "
        f"(delta={delta:.6f}) | plot -> {out_path}",
        flush=True,
    )


def plot_system_resources(history, out_path, device_index=0):
    """
    Plot RAM and GPU utilisation vs global step; log PNGs to W&B.
    """
    steps = history.get("_steps", [])
    if not steps:
        return

    series_specs = [
        ("system/ram_used_gb", "Host RAM used (GB)", "system/ram_plot"),
        ("system/ram_percent", "Host RAM (%)", "system/ram_percent_plot"),
        ("system/process_rss_gb", "Process RSS (GB)", "system/process_rss_plot"),
        ("system/gpu_utilization_percent", "GPU utilisation (%)", "system/gpu_util_plot"),
        ("system/gpu_memory_used_gb", "GPU memory used (GB)", "system/gpu_memory_used_plot"),
        ("system/gpu_memory_allocated_gb", "GPU memory allocated (GB)", "system/gpu_memory_allocated_plot"),
        ("system/gpu_memory_utilization_percent", "GPU memory util (%)", "system/gpu_memory_util_plot"),
    ]

    available = [
        (key, ylabel, wandb_key)
        for key, ylabel, wandb_key in series_specs
        if history.get(key) and any(v == v for v in history[key])
    ]
    if not available:
        return

    n_plots = len(available)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    wandb_images = {}
    for ax, (key, ylabel, wandb_key) in zip(axes, available):
        values = [v for v in history[key] if v == v]  # drop NaN
        plot_steps = steps[-len(values):] if len(values) < len(steps) else steps
        if not values:
            continue
        ax.plot(plot_steps, values, marker="o", markersize=2, linewidth=1.2)
        ax.set_xlabel("Global step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)

        single_fig, single_ax = plt.subplots(figsize=(8, 4))
        single_ax.plot(plot_steps, values, marker="o", markersize=3, linewidth=1.5)
        single_ax.set_xlabel("Global step")
        single_ax.set_ylabel(ylabel)
        single_ax.set_title(ylabel)
        single_ax.grid(True, alpha=0.3)
        single_path = out_path.replace(".png", f"_{wandb_key.split('/')[-1]}.png")
        single_fig.tight_layout()
        single_fig.savefig(single_path, dpi=150)
        plt.close(single_fig)
        wandb_images[wandb_key] = wandb.Image(single_path, caption=ylabel)

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(f"System resources (GPU index {device_index})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    summary = {
        "system/ram_used_gb_max": max(history["system/ram_used_gb"]) if history.get("system/ram_used_gb") else None,
        "system/ram_percent_max": max(history["system/ram_percent"]) if history.get("system/ram_percent") else None,
        "system/gpu_utilization_percent_max": (
            max(history["system/gpu_utilization_percent"])
            if history.get("system/gpu_utilization_percent") else None
        ),
    }
    summary = {k: v for k, v in summary.items() if v is not None}

    wandb.log({
        "system/resources_combined_plot": wandb.Image(out_path, caption="RAM & GPU over training"),
        **wandb_images,
        **summary,
    })
    print(f"[system] Resource plots -> {out_path}", flush=True)

# -----------------------------------------------------------------------------
# 6. PyTorch datasets
# -----------------------------------------------------------------------------

class ImageFitting(Dataset):
    def __init__(self, img, H, D, W):
        super().__init__()
        # img = get_cameraman_tensor(sidelength)
        img = img
        self.pixels = img.view(-1, 1)
        self.coords = create_3d_coords(H, D, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


class DataFeeding(Dataset):
    def __init__(self, img, H, D, W):
        super().__init__()
        # img = get_cameraman_tensor(sidelength)
        img = img
        self.pixels = img.view(-1, 1)


# def visualize_neuron_activation(activation, neuron_id, volume_shape=(64, 64, 64), slice_axis=2, slice_idx=None):
#     """
#     activation: torch.Tensor [1, N, 256]
#     neuron_id: which neuron to visualize
#     volume_shape: original 3D patch shape
#     slice_axis: 0, 1, or 2
#     """
#
#     D, H, W = volume_shape
#
#     act = activation[0, :, neuron_id].detach().cpu()
#     act_vol = act.view(D, H, W)
#
#     if slice_idx is None:
#         slice_idx = volume_shape[slice_axis] // 2
#
#     if slice_axis == 0:
#         slice_img = act_vol[slice_idx, :, :]
#     elif slice_axis == 1:
#         slice_img = act_vol[:, slice_idx, :]
#     elif slice_axis == 2:
#         slice_img = act_vol[:, :, slice_idx]
#     else:
#         raise ValueError("slice_axis must be 0, 1, or 2")
#
#     plt.figure(figsize=(6, 6))
#     plt.imshow(slice_img, cmap="viridis")
#     plt.title(f"Neuron {neuron_id} Activation | Axis {slice_axis}, Slice {slice_idx}")
#     plt.colorbar()
#     plt.axis("off")
#     plt.show()


class SIRENPatchDataset(Dataset):
    def __init__(self, patches, normalize=True):
        self.patches = patches
        self.normalize = normalize

        # assume all patches have same shape
        sample_patch = patches[0]["patch"]
        self.H, self.D, self.W = sample_patch.shape

        # same coordinate grid for all patches
        self.coords = create_3d_coords(self.H, self.D, self.W)  # [N, 3]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]["patch"].float()  # [H, D, W]

        if self.normalize:
            # fixed CBCT range normalization: [-1000, 5264] -> [-1, 1]
            patch = (patch + 1000) / 6264
            patch = patch * 2 - 1

        pixels = patch.reshape(-1, 1)  # [N, 1]

        return self.coords, pixels


# -----------------------------------------------------------------------------
# 7. Visualization utilities (W&B; TensorBoard disabled)
# -----------------------------------------------------------------------------

# def visualize_neuron_activation_tensorboard(
#     writer: SummaryWriter,
#     activation: torch.Tensor,
#     neuron_id: int,
#     global_step: int,
#     volume_shape=(64, 64, 64),
#     slice_axis=2,
#     slice_idx=None,
#     tag_prefix="NeuronActivation",
# ):
#     """
#     activation: [1, N, hidden_features]
#     Logs one neuron's activation slice to TensorBoard.
#     """
#
#     D, H, W = volume_shape
#
#     act = activation[0, :, neuron_id].detach().cpu()
#     act_vol = act.view(D, H, W)
#
#     if slice_idx is None:
#         slice_idx = volume_shape[slice_axis] // 2
#
#     if slice_axis == 0:
#         slice_img = act_vol[slice_idx, :, :]
#     elif slice_axis == 1:
#         slice_img = act_vol[:, slice_idx, :]
#     elif slice_axis == 2:
#         slice_img = act_vol[:, :, slice_idx]
#     else:
#         raise ValueError("slice_axis must be 0, 1, or 2")
#
#     slice_img = slice_img.float()
#     slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
#     slice_img = slice_img.unsqueeze(0)
#
#     tag = f"{tag_prefix}/neuron_{neuron_id}_axis_{slice_axis}_slice_{slice_idx}"
#     writer.add_image(tag, slice_img, global_step)


def _resolve_activation_slice_idx(volume_shape, slice_axis, slice_idx):
    """Map slice_idx to a valid index for the current patch (supports None / negative)."""
    axis_size = int(volume_shape[slice_axis])
    if slice_idx is None:
        return axis_size // 2
    idx = int(slice_idx)
    if idx < 0:
        idx = axis_size + idx
    return max(0, min(idx, axis_size - 1))


def _activation_slice_2d(
    activation: torch.Tensor,
    neuron_id: int,
    volume_shape,
    slice_axis=2,
    slice_idx=None,
):
    """
    activation: [1, N, hidden_features] -> normalised 2D slice [H, W] in [0, 1].
    """
    D, H, W = volume_shape

    act = activation[0, :, neuron_id].detach().cpu()
    act_vol = act.view(D, H, W)

    slice_idx = _resolve_activation_slice_idx(volume_shape, slice_axis, slice_idx)

    if slice_axis == 0:
        slice_img = act_vol[slice_idx, :, :]
    elif slice_axis == 1:
        slice_img = act_vol[:, slice_idx, :]
    elif slice_axis == 2:
        slice_img = act_vol[:, :, slice_idx]
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")

    slice_img = slice_img.float()
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    return slice_img, int(slice_idx)


def _make_neuron_montage(slices, ncols=16):
    """Tile per-neuron 2D slices into one grid image."""
    n = len(slices)
    nrows = (n + ncols - 1) // ncols
    h, w = slices[0].shape
    grid = np.zeros((nrows * h, ncols * w), dtype=np.float32)
    for i, sl in enumerate(slices):
        r, c = divmod(i, ncols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = sl
    return grid


def _activation_slice_to_uint8(slice_arr: np.ndarray) -> np.ndarray:
    """Convert normalised [0, 1] activation slice to uint8 grayscale."""
    return (np.clip(slice_arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_activation_map_png(slice_arr: np.ndarray, path: str) -> str:
    """Write one neuron activation map to disk as a PNG image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(_activation_slice_to_uint8(slice_arr), mode="L").save(path)
    return path


def log_all_neuron_activations_wandb(
    activation: torch.Tensor,
    patch_idx: int,
    step: int,
    global_step: int,
    volume_shape,
    neuron_ids,
    slice_axis=2,
    slice_idx=None,
    hidden_features=256,
    maps_dir="activation_maps",
    log_wandb_images=False,
    wandb_batch_size=32,
    log_montage=True,
    montage_ncols=16,
    log_artifact=False,
):
    """
    Save activation maps as PNG files for a subset of neurons (e.g. 50 of 256).

    Avoids wandb.Table / Histogram (they create large staging copies and can
    trigger 'No space left on device'). Optional W&B upload: montage and/or
    per-neuron images (off by default to save disk).
    """
    num_neurons = activation.shape[-1]
    if num_neurons != hidden_features:
        hidden_features = num_neurons

    neuron_ids = sorted({int(n) for n in neuron_ids if 0 <= int(n) < hidden_features})
    if not neuron_ids:
        raise ValueError("neuron_ids is empty or out of range")

    snapshot_key = f"patch_{patch_idx:04d}_step_{step:04d}"
    out_dir = os.path.join(maps_dir, f"patch_{patch_idx:04d}", f"step_{step:04d}")
    os.makedirs(out_dir, exist_ok=True)

    slice_arrays = []
    image_paths = []
    logged_neuron_ids = []
    slice_idx_used = slice_idx

    for neuron_id in neuron_ids:
        slice_img, slice_idx_used = _activation_slice_2d(
            activation,
            neuron_id,
            volume_shape,
            slice_axis=slice_axis,
            slice_idx=slice_idx,
        )
        arr = slice_img.numpy()
        slice_arrays.append(arr)
        logged_neuron_ids.append(neuron_id)
        png_path = os.path.join(out_dir, f"neuron_{neuron_id:03d}.png")
        _save_activation_map_png(arr, png_path)
        image_paths.append(png_path)

    print(
        f"[activations] Saved {len(logged_neuron_ids)}/{hidden_features} PNG maps "
        f"-> {out_dir}",
        flush=True,
    )

    caption_base = (
        f"patch={patch_idx} step={step} "
        f"| axis={slice_axis} slice={slice_idx_used}"
    )

    if log_montage:
        montage = _make_neuron_montage(slice_arrays, ncols=montage_ncols)
        wandb.log(
            {
                f"activations/{snapshot_key}/montage_{len(logged_neuron_ids)}_neurons": wandb.Image(
                    montage,
                    caption=f"Overview ({len(logged_neuron_ids)} neurons) | {caption_base}",
                    mode="L",
                ),
            },
            step=global_step,
        )

    if log_wandb_images:
        batch_size = max(1, int(wandb_batch_size))
        for batch_start in range(0, len(logged_neuron_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(logged_neuron_ids))
            batch_payload = {}
            for i in range(batch_start, batch_end):
                neuron_id = logged_neuron_ids[i]
                batch_payload[
                    f"activations/{snapshot_key}/neuron_{neuron_id:03d}"
                ] = wandb.Image(
                    image_paths[i],
                    caption=f"neuron={neuron_id} | {caption_base}",
                    mode="L",
                )
            wandb.log(batch_payload, step=global_step)

    if log_artifact:
        artifact = wandb.Artifact(
            name=f"activation-maps-{snapshot_key}",
            type="activation-maps",
            description=(
                f"{len(logged_neuron_ids)} neuron activation PNGs | "
                f"patch={patch_idx} step={step}"
            ),
        )
        artifact.add_dir(out_dir)
        wandb.log_artifact(artifact)


# -----------------------------------------------------------------------------
# 8. LPIPS metric helpers
# -----------------------------------------------------------------------------

def prepare_lpips_2d(slice_2d):
    """
    Input:  [H, W]
    Output: [1, 3, H, W] in [-1, 1]
    """
    if not torch.is_tensor(slice_2d):
        slice_2d = torch.from_numpy(slice_2d)

    x = slice_2d.float()

    # normalize slice to [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # convert [0, 1] to [-1, 1]
    x = x * 2 - 1

    # [H, W] -> [1, 1, H, W]
    x = x.unsqueeze(0).unsqueeze(0)

    # LPIPS needs 3 channels: [1, 1, H, W] -> [1, 3, H, W]
    x = x.repeat(1, 3, 1, 1)
    return x.to(device)


def compute_lpips_for_slice(recon_slice, ori_slice, view_name):

    recon_t = prepare_lpips_2d(recon_slice)
    ori_t = prepare_lpips_2d(ori_slice)

    with torch.no_grad():
        d_alex = loss_fn_alex(recon_t, ori_t).item()
        d_vgg = loss_fn_vgg(recon_t, ori_t).item()

    print(f"{view_name} | LPIPS Alex: {d_alex:.6f} | LPIPS VGG: {d_vgg:.6f}")

    wandb.log({
        f"metrics/lpips_alex/{view_name}": d_alex,
        f"metrics/lpips_vgg/{view_name}": d_vgg,
    })

    return {
        "view": view_name,
        "alex": d_alex,
        "vgg": d_vgg,
        "shape": tuple(recon_t.shape)
    }


# -----------------------------------------------------------------------------
# 9. Training and evaluation (script entry)
# -----------------------------------------------------------------------------

input_path = cfg.input_data_name
img = np.load(input_path)
img = torch.from_numpy(img)
img = img_normalisation(img)
print(img.shape)
ori_shape = img.shape

wandb.config.update({"volume_shape": list(ori_shape)}, allow_val_change=True)

patches, _ = extract_patches_monai(img, patch_size, stride)
num_patches = len(patches)  # determined by MONAI dense_patch_slices on the volume
wandb.config.update({"num_patches": num_patches}, allow_val_change=True)
print(
    f"MONAI patches: {num_patches} "
    f"(volume={ori_shape}, patch_size={patch_size}, stride={stride})"
)

global_noise_reference = build_global_noise_reference(
    img,
    a=cfg.noise_model_a,
    b=cfg.noise_model_b,
    hu_bin_width=cfg.noise_model_hu_bin_width,
    overlap_ratio=cfg.noise_model_overlap_ratio,
    bins=cfg.noise_model_bins,
    seed=cfg.noise_model_seed,
)
wandb.config.update(
    {
        "noise_reference_scope": "global_volume",
        "noise_global_hu_min": global_noise_reference["hu_min"],
        "noise_global_hu_max": global_noise_reference["hu_max"],
        "noise_global_hu_range": global_noise_reference["hu_range"],
        "noise_global_offset_min": global_noise_reference["offset_min"],
        "noise_global_offset_max": global_noise_reference["offset_max"],
        "noise_global_reference_noise_mean": global_noise_reference["reference_noise_mean"],
        "noise_global_reference_noise_std": global_noise_reference["reference_noise_std"],
        "noise_global_reference_residual_std": global_noise_reference["reference_residual_std"],
    },
    allow_val_change=True,
)
wandb.log({
    "noise/global_hu_min": global_noise_reference["hu_min"],
    "noise/global_hu_max": global_noise_reference["hu_max"],
    "noise/global_hu_range": global_noise_reference["hu_range"],
    "noise/global_offset_min": global_noise_reference["offset_min"],
    "noise/global_offset_max": global_noise_reference["offset_max"],
    "noise/global_reference_noise_mean": global_noise_reference["reference_noise_mean"],
    "noise/global_reference_noise_std": global_noise_reference["reference_noise_std"],
    "noise/global_reference_residual_std": global_noise_reference["reference_residual_std"],
})
print(
    "Global noise reference built from full volume | "
    f"HU range={global_noise_reference['hu_range']:.3f} | "
    f"noise std={global_noise_reference['reference_noise_std']:.6f}",
    flush=True,
)

img_siren = Siren(
    in_features=3,
    out_features=1,
    hidden_features=cfg.hidden_features,
    hidden_layers=cfg.hidden_layers,
    outermost_linear=cfg.outermost_linear,
    first_omega_0=cfg.first_omega_0,
    hidden_omega_0=cfg.hidden_omega_0,
)
img_siren.cuda()

omega_initial_values = collect_initial_omega_values(img_siren)
wandb.config.update(
    {
        "first_omega_0_config": float(cfg.first_omega_0),
        "hidden_omega_0_config": float(cfg.hidden_omega_0),
        **omega_initial_values,
    },
    allow_val_change=True,
)
print(
    f"SIREN omega init | first_omega_0={cfg.first_omega_0} "
    f"hidden_omega_0={cfg.hidden_omega_0} | per-layer: {omega_initial_values}",
    flush=True,
)

H, W, D = patches[0]['patch'].shape
total_steps = cfg.total_steps
steps_til_summary = cfg.steps_til_summary

optim = build_optimizer(img_siren, cfg.optimizer, cfg.learning_rate)

tracked_omega_layer_idx, tracked_omega_layer = pick_tracked_omega_layer(
    img_siren, seed=cfg.omega_track_seed
)
omega_history = []
system_history = new_system_history()
psutil.cpu_percent(interval=None)  # prime CPU counter

wandb.config.update(
    {
        "omega_tracked_layer_idx": tracked_omega_layer_idx,
        "omega_tracked_is_first_layer": tracked_omega_layer.is_first,
        "omega_tracked_in_features": tracked_omega_layer.in_features,
        "omega_tracked_out_features": tracked_omega_layer.linear.out_features,
        "omega_tracked_initial": tracked_omega_layer.omega_0.detach().item(),
        "omega_num_sine_layers": len(get_sine_layers(img_siren)),
    },
    allow_val_change=True,
)
print(
    f"Tracking trainable omega_0 on SineLayer {tracked_omega_layer_idx} "
    f"({'first' if tracked_omega_layer.is_first else 'hidden'}, "
    f"out_features={tracked_omega_layer.linear.out_features}) | "
    f"initial={tracked_omega_layer.omega_0.item():.6f}",
    flush=True,
)

snapshot_patch_set, snapshot_steps = sample_activation_snapshots(
    num_patches=num_patches,
    total_steps=total_steps,
    num_snapshot_patches=cfg.activation_snapshot_num_patches,
    seed=cfg.activation_log_seed,
)
activation_neuron_ids = sample_activation_neuron_ids(
    total_neurons=cfg.hidden_features,
    num_neurons_to_log=cfg.activation_num_neurons_to_log,
    seed=cfg.activation_log_seed + 1,
)
wandb.config.update(
    {
        "activation_snapshot_patch_indices": sorted(snapshot_patch_set),
        "activation_snapshot_steps": {str(k): v for k, v in snapshot_steps.items()},
        "activation_neuron_ids": activation_neuron_ids,
        "activation_num_neurons_logged": len(activation_neuron_ids),
    },
    allow_val_change=True,
)
print(
    f"Activation snapshots: {len(snapshot_patch_set)} random patches, "
    f"one random step each -> {snapshot_steps} | "
    f"logging {len(activation_neuron_ids)} neurons: {activation_neuron_ids}",
)

global_start = time.perf_counter()

processes_patches = []
activation_snapshot_js_records = []

patch_shape = patches[0]['patch'].shape

# writer = SummaryWriter("runs/siren_activation_debug")

for patch_idx, img_patch in enumerate(patches):
    patch = img_patch['patch']
    H, D, W = patch.shape

    Img_fitting = ImageFitting(patch, H, D, W)
    dataloader = DataLoader(Img_fitting, batch_size=1, pin_memory=True, num_workers=1)

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    patch_noise = PatchNoiseModel(
        a=cfg.noise_model_a,
        b=cfg.noise_model_b,
        hu_bin_width=cfg.noise_model_hu_bin_width,
        overlap_ratio=cfg.noise_model_overlap_ratio,
        bins=cfg.noise_model_bins,
    )
    patch_noise.setup_from_global(
        ground_truth,
        (H, D, W),
        img_patch['slice'],
        global_noise_reference,
        device=ground_truth.device,
    )
    wandb.log(
        {
            f"patch/{patch_idx:04d}/noise_setup_hu_min": patch_noise.setup_metrics["noise/setup_hu_min"],
            f"patch/{patch_idx:04d}/noise_setup_hu_max": patch_noise.setup_metrics["noise/setup_hu_max"],
            f"patch/{patch_idx:04d}/noise_setup_hu_range": patch_noise.setup_metrics["noise/setup_hu_range"],
            f"patch/{patch_idx:04d}/noise_setup_offset_min": patch_noise.setup_metrics["noise/setup_offset_min"],
            f"patch/{patch_idx:04d}/noise_setup_offset_max": patch_noise.setup_metrics["noise/setup_offset_max"],
            f"patch/{patch_idx:04d}/reference_noise_mean": patch_noise.setup_metrics["noise/setup_reference_noise_mean"],
            f"patch/{patch_idx:04d}/reference_noise_std": patch_noise.setup_metrics["noise/setup_reference_noise_std"],
            f"patch/{patch_idx:04d}/reference_residual_std": patch_noise.setup_metrics["noise/setup_reference_residual_std"],
        },
        step=patch_idx * total_steps,
    )

    print(f"\n=== Patch {patch_idx+1}/{len(patches)} ===")
    print(f"model_input: {model_input.shape} | ground_truth: {ground_truth.shape}")

    patch_start = time.perf_counter()

    for step in range(total_steps):

        torch.cuda.synchronize()
        step_start = time.perf_counter()

        model_output, coords = img_siren(model_input)

        output_vol = model_output[0].view(H, D, W)
        gt_vol = ground_truth[0].view(H, D, W)

        gradient_model = torch.gradient(output_vol, spacing=1)
        gradient_gt = torch.gradient(gt_vol, spacing=1)

        gradient_model = torch.stack(gradient_model, dim=0)
        gradient_gt = torch.stack(gradient_gt, dim=0)

        grad_diff = gradient_model - gradient_gt

        grad_diff_mag = torch.sqrt(torch.sum(grad_diff ** 2, dim=0) + 1e-8)

        gradient_loss = grad_diff_mag.mean()
        mse_loss = ((model_output - ground_truth) ** 2).mean()
        tv_loss = tv_loss_3d(output_vol)

        zero_loss = torch.zeros((), dtype=model_output.dtype, device=model_output.device)
        noise_mse_loss = (
            patch_noise.noise_mse_loss(model_output)
            if float(cfg.loss_noise_mse_weight) != 0.0
            else zero_loss
        )
        noise_js_loss = (
            patch_noise.noise_js_loss(model_output)
            if float(cfg.loss_noise_js_weight) != 0.0
            else zero_loss
        )
        residual_noise_js_loss = patch_noise.residual_noise_js_loss(
            model_output, ground_truth
        )
        residual_patch_js = patch_noise.residual_patch_js_numpy(
            model_output, ground_truth
        )
        legacy_output_patch_js = patch_noise.patch_js_numpy(model_output)
        residual = ground_truth - model_output
        gt_std = ground_truth.detach().std()
        output_std = model_output.detach().std()
        residual_std = residual.detach().std()
        residual_mean_abs = residual.detach().abs().mean()
        residual_energy_ratio = (
            residual.detach().pow(2).mean()
            / (ground_truth.detach().pow(2).mean() + 1e-8)
        )

        loss = (
            cfg.loss_mse_weight * mse_loss
            + cfg.loss_grad_weight * gradient_loss
            + cfg.loss_tv_weight * tv_loss
            + cfg.loss_noise_mse_weight * noise_mse_loss
            + cfg.loss_noise_js_weight * noise_js_loss
            + cfg.loss_residual_noise_js_weight * residual_noise_js_loss
        )

        global_step = patch_idx * total_steps + step

        # Activation maps for a random subset of neurons at one random step per patch
        if patch_idx in snapshot_patch_set and step == snapshot_steps[patch_idx]:
            layer_act = LayerActivation(img_siren, img_siren.net[3])
            attribution = layer_act.attribute(model_input)
            print(
                f"[W&B] Logging {len(activation_neuron_ids)} neuron activations | "
                f"patch={patch_idx} step={step}",
                flush=True,
            )
            log_all_neuron_activations_wandb(
                activation=attribution,
                patch_idx=patch_idx,
                step=step,
                global_step=global_step,
                volume_shape=patch_shape,
                neuron_ids=activation_neuron_ids,
                slice_axis=cfg.activation_slice_axis,
                slice_idx=cfg.activation_slice_idx,
                hidden_features=cfg.hidden_features,
                maps_dir=cfg.activation_maps_dir,
                log_wandb_images=cfg.activation_log_wandb_images,
                wandb_batch_size=cfg.activation_wandb_batch_size,
                log_montage=cfg.activation_log_montage,
                montage_ncols=16,
                log_artifact=cfg.activation_log_artifact,
            )
            if cfg.activation_snapshot_log_js:
                activation_snapshot_js_records.append(
                    {
                        "patch_idx": int(patch_idx),
                        "step": int(step),
                        "global_step": int(global_step),
                        "output_js": float(legacy_output_patch_js),
                        "residual_js": float(residual_patch_js),
                        "residual_noise_js_loss": float(residual_noise_js_loss.item()),
                        "residual_std": float(residual_std.item()),
                        "output_to_input_std_ratio": float(
                            (output_std / (gt_std + 1e-8)).item()
                        ),
                    }
                )
                wandb.log(
                    {
                        "activation_snapshot_js/output_js": float(legacy_output_patch_js),
                        "activation_snapshot_js/residual_js": float(residual_patch_js),
                        "activation_snapshot_js/residual_noise_js_loss": float(
                            residual_noise_js_loss.item()
                        ),
                        "activation_snapshot_js/patch_idx": int(patch_idx),
                        "activation_snapshot_js/patch_step": int(step),
                        f"activation_snapshot_js/patch_{patch_idx:04d}_output_js": float(
                            legacy_output_patch_js
                        ),
                        f"activation_snapshot_js/patch_{patch_idx:04d}_residual_js": float(
                            residual_patch_js
                        ),
                    },
                    step=global_step,
                )

        torch.cuda.synchronize()
        step_end = time.perf_counter()

        elapsed_global = step_end - global_start
        elapsed_patch = step_end - patch_start
        step_time = step_end - step_start

        log_dict = {
            "train/loss": loss.item(),
            "train/loss_mse": mse_loss.item(),
            "train/loss_gradient": gradient_loss.item(),
            "train/loss_tv": tv_loss.item(),
            "train/loss_noise_mse": noise_mse_loss.item(),
            "train/loss_noise_js": noise_js_loss.item(),
            "train/loss_residual_noise_js": residual_noise_js_loss.item(),
            "train/legacy_output_noise_js": legacy_output_patch_js,
            "train/weighted_loss_mse": (cfg.loss_mse_weight * mse_loss).item(),
            "train/weighted_loss_gradient": (cfg.loss_grad_weight * gradient_loss).item(),
            "train/weighted_loss_tv": (cfg.loss_tv_weight * tv_loss).item(),
            "train/weighted_loss_noise_mse": (cfg.loss_noise_mse_weight * noise_mse_loss).item(),
            "train/weighted_loss_noise_js": (cfg.loss_noise_js_weight * noise_js_loss).item(),
            "train/weighted_loss_residual_noise_js": (
                cfg.loss_residual_noise_js_weight * residual_noise_js_loss
            ).item(),
            "denoise/input_std": gt_std.item(),
            "denoise/output_std": output_std.item(),
            "denoise/residual_std": residual_std.item(),
            "denoise/residual_mean_abs": residual_mean_abs.item(),
            "denoise/residual_energy_ratio": residual_energy_ratio.item(),
            "denoise/output_to_input_std_ratio": (
                output_std / (gt_std + 1e-8)
            ).item(),
            f"noise/patch_{patch_idx:04d}_js": legacy_output_patch_js,
            f"noise/patch_{patch_idx:04d}_residual_js": residual_patch_js,
            "train/patch_idx": patch_idx,
            "train/patch_step": step,
            "train/step_time_s": step_time,
            "train/patch_time_s": elapsed_patch,
            "train/total_time_s": elapsed_global,
        }

        hw_metrics = get_hardware_metrics(device_index=cfg.gpu_device_index)
        log_dict.update(hw_metrics)
        append_system_history(system_history, hw_metrics, global_step)

        optim.zero_grad()
        loss.backward()
        optim.step()

        omega_val = get_sine_layers(img_siren)[tracked_omega_layer_idx].omega_0.item()
        omega_history.append((global_step, omega_val))
        log_dict.update(collect_omega_metrics(img_siren, tracked_omega_layer_idx))
        log_dict["omega/tracked_value_post_step"] = omega_val

        if step % steps_til_summary == 0:
            print(
                f"[Patch {patch_idx+1:03d}] "
                f"Step {step:04d}/{total_steps} | "
                f"Loss: {loss.item():.6f} "
                f"(mse={mse_loss.item():.6f}, grad={gradient_loss.item():.6f}, "
                f"tv={tv_loss.item():.6f}, "
                f"noise_mse={noise_mse_loss.item():.6f}, noise_js={noise_js_loss.item():.6f}, "
                f"residual_js={residual_patch_js:.6f}) | "
                f"omega(L{tracked_omega_layer_idx})={omega_val:.6f} | "
                f"StepTime: {step_time:.4f}s | "
                f"PatchTime: {elapsed_patch:.2f}s | "
                f"TotalTime: {elapsed_global:.2f}s",
                flush=True
            )

        wandb.log(log_dict, step=global_step)

    pred_dict = {
                "patch": model_output.squeeze(-1).reshape(patch_shape),
                "slice": img_patch['slice']
            }

    processes_patches.append(pred_dict)
    patch_end = time.perf_counter()
    final_patch_js = patch_noise.residual_patch_js_numpy(model_output, ground_truth)
    final_legacy_output_patch_js = patch_noise.patch_js_numpy(model_output)
    wandb.log({
        f"noise/patch_{patch_idx:04d}_js_final": final_legacy_output_patch_js,
        f"noise/patch_{patch_idx:04d}_js": final_legacy_output_patch_js,
        f"noise/patch_{patch_idx:04d}_residual_js_final": final_patch_js,
        f"noise/patch_{patch_idx:04d}_residual_js": final_patch_js,
    })
    print(
        f"--- Patch {patch_idx+1} completed in {patch_end - patch_start:.2f}s | "
        f"final residual JS={final_patch_js:.6f} ---",
        flush=True,
    )


# writer.flush()

training_time_s = time.perf_counter() - global_start
print(f"\nTraining completed in {training_time_s:.2f}s")

if activation_snapshot_js_records:
    activation_js_table = wandb.Table(
        columns=[
            "patch_idx",
            "step",
            "global_step",
            "output_js",
            "residual_js",
            "residual_noise_js_loss",
            "residual_std",
            "output_to_input_std_ratio",
        ]
    )
    for row in activation_snapshot_js_records:
        activation_js_table.add_data(
            row["patch_idx"],
            row["step"],
            row["global_step"],
            row["output_js"],
            row["residual_js"],
            row["residual_noise_js_loss"],
            row["residual_std"],
            row["output_to_input_std_ratio"],
        )

    activation_residual_js_values = [
        row["residual_js"] for row in activation_snapshot_js_records
    ]
    activation_output_js_values = [
        row["output_js"] for row in activation_snapshot_js_records
    ]
    wandb.log({
        "activation_snapshot_js/table": activation_js_table,
        "activation_snapshot_js/num_records": len(activation_snapshot_js_records),
        "activation_snapshot_js/residual_js_mean": float(np.mean(activation_residual_js_values)),
        "activation_snapshot_js/residual_js_std": float(np.std(activation_residual_js_values)),
        "activation_snapshot_js/output_js_mean": float(np.mean(activation_output_js_values)),
        "activation_snapshot_js/output_js_std": float(np.std(activation_output_js_values)),
    })

plot_omega_tracking(
    omega_history,
    tracked_omega_layer_idx,
    tracked_omega_layer.is_first,
    cfg.omega_plot_path,
)
wandb.save(cfg.omega_plot_path)

plot_system_resources(
    system_history,
    cfg.system_plot_path,
    device_index=cfg.gpu_device_index,
)
wandb.save(cfg.system_plot_path)

wandb.log({
    "train/total_training_time_s": training_time_s,
    "omega/config_first_omega_0": float(cfg.first_omega_0),
    "omega/config_hidden_omega_0": float(cfg.hidden_omega_0),
})

# Save model checkpoint
checkpoint_path = "checkpoint.pth"
torch.save({
    "model_state_dict": img_siren.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
    "loss": loss.item(),
    "wandb_config": dict(cfg),
}, checkpoint_path)

wandb.save(checkpoint_path)

recon_img, count = reconstruct_patches_monai(processes_patches, ori_shape)

np.save(
    "reconstructed_volume.npy",
    recon_img.detach().cpu().numpy()
)

recon_np = recon_img.detach().cpu().numpy()
img_np = img.detach().cpu().numpy()

print("======SSIM Score========")
print("recon min:", recon_np.min())
print("recon max:", recon_np.max())

print("img min:", img_np.min())
print("img max:", img_np.max())

# Calculate SSIM
score, diff = ssim(recon_np, img_np, data_range=2, full=True, channel_axis=2)
print("SSIM Score:", score)

wandb.log({
    "metrics/ssim": score,
    "metrics/recon_min": float(recon_np.min()),
    "metrics/recon_max": float(recon_np.max()),
    "metrics/img_min": float(img_np.min()),
    "metrics/img_max": float(img_np.max()),
})

print("======LPIPS Score=======")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net="alex").to(device)
loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

# -------- coronal view: [:, 115, :] --------
recon_coronal = recon_img[:, 115, :]
ori_coronal = img[:, 115, :]

# -------- sagittal view: [:, :, 115] --------
recon_sagittal = recon_img[:, :, 115]
ori_sagittal = img[:, :, 115]

# -------- axial view: [100, :, :] --------
recon_axial = recon_img[100, :, :]
ori_axial = img[100, :, :]   # fixed typo: img[100. :, :] -> img[100, :, :]

lpips_results = []
lpips_results.append(
    compute_lpips_for_slice(recon_coronal, ori_coronal, "Coronal [:,115,:]")
)

lpips_results.append(
    compute_lpips_for_slice(recon_sagittal, ori_sagittal, "Sagittal [:,:,115]")
)

lpips_results.append(
    compute_lpips_for_slice(recon_axial, ori_axial, "Axial [100,:,:]")
)

print("\n====== Summary ======")
for r in lpips_results:
    print(
        f"{r['view']} | "
        f"Input shape: {r['shape']} | "
        f"Alex: {r['alex']:.6f} | "
        f"VGG: {r['vgg']:.6f}"
    )
wandb.finish()
