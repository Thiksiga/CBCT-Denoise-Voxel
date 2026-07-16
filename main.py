import os
import random
import time
import psutil
import torch
import wandb
import numpy as np
import lpips
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from captum.attr import LayerActivation

# Local module imports
from models import Siren, get_sine_layers, pick_tracked_omega_layer, collect_omega_metrics, collect_initial_omega_values
from image_processing import extract_patches_monai, reconstruct_patches_monai, img_normalisation, ImageFitting
from losses import tv_loss_3d, build_global_noise_reference, PatchNoiseModel

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

_NVML_INITIALIZED = False
_DEFAULT_INPUT = r"/projects1/Toothfairy/ToothFairy_Dataset/Dataset/Dataset/P10/data.npy"

WANDB_API_KEY = (
"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"
)
os.environ["WANDB_API_KEY"] = WANDB_API_KEY
wandb.login(key=WANDB_API_KEY, relogin=True)

# -----------------------------------------------------------------------------
# Configuration and Hardware Tracking
# -----------------------------------------------------------------------------
wandb.init(
project="zero-shot-inr",
name=f"SIREN_zero_shot_2_MSE_gradLoss_{os.path.splitext(os.path.basename(_DEFAULT_INPUT))[0]}",
config={
    "architecturesere": "SIREN",
    "training_mode": "zero_shot",
    "hidden_features": 256,
    "hidden_layers": 3,
    "outermost_linear": True,
    "first_omega_0": 60,
    "hidden_omega_0": 60,
    "input_data_name": _DEFAULT_INPUT,
    "patch_size": [32, 32, 32],
    "stride": [16, 16, 16],
    "learning_rate": 1e-5,
    "optimizer": "Adam",
    "loss_mse_weight": 1.0,
    "loss_grad_weight": 1.0,
    "loss_noise_mse_weight": 0.0,
    "loss_noise_js_weight": 0.0,
    "loss_tv_weight": 1e-5,
    "loss_residual_noise_js_weight": 0.001,
    "noise_model_a": 1.0,
    "noise_model_b": 0.01,
    "noise_model_hu_bin_width": 200,
    "noise_model_overlap_ratio": 0.05,
    "noise_model_bins": 100,
    "noise_model_seed": 42,
    "total_steps": 50,
    "steps_til_summary": 10,
    "gpu_device_index": 0,
    "activation_snapshot_num_patches": 50,
    "activation_num_neurons_to_log": 50,
    "activation_log_seed": 42,
    "activation_slice_axis": 2,
    "activation_slice_idx": None,
    "activation_maps_dir": "activation_maps",
    "activation_log_wandb_images": False,
    "activation_log_montage": True,
    "activation_wandb_batch_size": 32,
    "activation_log_artifact": False,
    "activation_snapshot_log_js": True,
    "omega_track_seed": 42,
    "omega_plot_path": "omega_tracked.png",
    "system_plot_path": "system_resources.png",
},
)
cfg = wandb.config

def build_optimizer(model, optimizer_name: str, learning_rate: float):
    name = optimizer_name.lower()
    if name == "adam": return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "adamw": return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if name == "sgd": return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def _init_nvml():
    global _NVML_INITIALIZED
    if _NVML_AVAILABLE and not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True

def get_hardware_metrics(device_index=0):
    metrics = {}
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    metrics["system/ram_used_gb"] = vm.used / (1024 ** 3)
    metrics["system/ram_percent"] = vm.percent
    metrics["system/process_rss_gb"] = proc.memory_info().rss / (1024 ** 3)
    metrics["system/cpu_utilization_percent"] = psutil.cpu_percent(interval=None)

    if torch.cuda.is_available():
        metrics["system/compute_device"] = f"cuda:{device_index}"
        metrics["system/gpu_utilization_percent"] = 0
        metrics["system/gpu_memory_used_gb"] = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device_index) / (1024 ** 3)

        if _NVML_AVAILABLE:
            _init_nvml()
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

    SYSTEM_HISTORY_KEYS = ("system/ram_used_gb", "system/ram_percent", "system/process_rss_gb", "system/gpu_utilization_percent", "system/gpu_memory_used_gb", "system/gpu_memory_allocated_gb", "system/gpu_memory_utilization_percent")
    def new_system_history(): return {key: [] for key in SYSTEM_HISTORY_KEYS} | {"_steps": []}
    def append_system_history(history, metrics, global_step):
        history["_steps"].append(global_step)
        for key in SYSTEM_HISTORY_KEYS: history[key].append(metrics.get(key, float("nan")))

def sample_activation_snapshots(num_patches, total_steps, num_snapshot_patches, seed):
    rng = random.Random(seed)
    n = min(int(num_snapshot_patches), num_patches)
    patch_indices = sorted(rng.sample(range(num_patches), n))
    step_by_patch = {pi: rng.randrange(total_steps) for pi in patch_indices}
    return set(patch_indices), step_by_patch

def sample_activation_neuron_ids(total_neurons, num_neurons_to_log, seed):
    rng = random.Random(seed)
    n = min(int(num_neurons_to_log), int(total_neurons))
    return sorted(rng.sample(range(int(total_neurons)), n))

# -----------------------------------------------------------------------------
# Visualization Utilities
# -----------------------------------------------------------------------------
def plot_omega_tracking(history, tracked_layer_idx, is_first_layer, out_path):
    if not history: return
    steps, values = zip(*history)
    initial, final = values[0], values[-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(initial, color="gray", linestyle="--", alpha=0.6)
    ax.set_title(f"Trainable omega_0 | tracked SineLayer {tracked_layer_idx}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    wandb.log({"omega/tracked_plot": wandb.Image(out_path)})

def plot_system_resources(history, out_path, device_index=0):
    steps = history.get("_steps", [])
    if not steps: return
    series_specs = [("system/ram_used_gb", "Host RAM used (GB)"), ("system/gpu_utilization_percent", "GPU utilisation (%)"), ("system/gpu_memory_used_gb", "GPU memory used (GB)")]
    available = [(key, ylabel) for key, ylabel in series_specs if history.get(key)]
    if not available: return

    fig, axes = plt.subplots(len(available), 1, figsize=(8, 4 * len(available)))
    axes = np.atleast_1d(axes).flatten()
    for ax, (key, ylabel) in zip(axes, available):
        values = history[key]
        ax.plot(steps, values, marker="o", linewidth=1.2)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    wandb.log({"system/resources_combined_plot": wandb.Image(out_path)})

# Activation Mapping
def _resolve_activation_slice_idx(volume_shape, slice_axis, slice_idx):
    axis_size = int(volume_shape[slice_axis])
    if slice_idx is None: return axis_size // 2
    idx = int(slice_idx)
    if idx < 0: idx = axis_size + idx
    return max(0, min(idx, axis_size - 1))

def _activation_slice_2d(activation, neuron_id, volume_shape, slice_axis=2, slice_idx=None):
    D, H, W = volume_shape
    act = activation[0, :, neuron_id].detach().cpu().view(D, H, W)
    slice_idx = _resolve_activation_slice_idx(volume_shape, slice_axis, slice_idx)
    slice_img = act[slice_idx, :, :] if slice_axis == 0 else (act[:, slice_idx, :] if slice_axis == 1 else act[:, :, slice_idx])
    slice_img = slice_img.float()
    return (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8), int(slice_idx)

def log_all_neuron_activations_wandb(activation, patch_idx, step, global_step, volume_shape, neuron_ids, slice_axis, slice_idx, hidden_features, maps_dir, log_montage=True):
    neuron_ids = [n for n in neuron_ids if 0 <= n < hidden_features]
    if not neuron_ids: return
    out_dir = os.path.join(maps_dir, f"patch_{patch_idx:04d}", f"step_{step:04d}")
    os.makedirs(out_dir, exist_ok=True)
    slice_arrays = []

    for neuron_id in neuron_ids:
        slice_img, _ = _activation_slice_2d(activation, neuron_id, volume_shape, slice_axis, slice_idx)
        arr = slice_img.numpy()
        slice_arrays.append(arr)
        png_path = os.path.join(out_dir, f"neuron_{neuron_id:03d}.png")
        Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(png_path)

    if log_montage:
        ncols = 16
        n = len(slice_arrays)
        nrows = (n + ncols - 1) // ncols
        h, w = slice_arrays[0].shape
        grid = np.zeros((nrows * h, ncols * w), dtype=np.float32)
        for i, sl in enumerate(slice_arrays):
            r, c = divmod(i, ncols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = sl
        wandb.log({f"activations/patch_{patch_idx:04d}_step_{step:04d}/montage": wandb.Image(grid, mode="L")}, step=global_step)

def prepare_lpips_2d(slice_2d, device):
    if not torch.is_tensor(slice_2d): slice_2d = torch.from_numpy(slice_2d)
    x = slice_2d.float()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 2 - 1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

def compute_lpips_for_slice(recon_slice, ori_slice, view_name, loss_fn_alex, loss_fn_vgg, device):
    recon_t = prepare_lpips_2d(recon_slice, device)
    ori_t = prepare_lpips_2d(ori_slice, device)
    with torch.no_grad():
        d_alex = loss_fn_alex(recon_t, ori_t).item()
        d_vgg = loss_fn_vgg(recon_t, ori_t).item()
    wandb.log({f"metrics/lpips_alex/{view_name}": d_alex, f"metrics/lpips_vgg/{view_name}": d_vgg})
    return {"view": view_name, "alex": d_alex, "vgg": d_vgg}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    img = np.load(cfg.input_data_name)
    img = torch.from_numpy(img)
    img = img_normalisation(img)
    ori_shape = img.shape
    wandb.config.update({"volume_shape": list(ori_shape)}, allow_val_change=True)

    patches, _ = extract_patches_monai(img, tuple(cfg.patch_size), tuple(cfg.stride))
    num_patches = len(patches)
    wandb.config.update({"num_patches": num_patches}, allow_val_change=True)

    global_noise_reference = build_global_noise_reference(
        img, a=cfg.noise_model_a, b=cfg.noise_model_b, hu_bin_width=cfg.noise_model_hu_bin_width, overlap_ratio=cfg.noise_model_overlap_ratio, bins=cfg.noise_model_bins, seed=cfg.noise_model_seed
    )

    img_siren = Siren(
        in_features=3, out_features=1, hidden_features=cfg.hidden_features, hidden_layers=cfg.hidden_layers, outermost_linear=cfg.outermost_linear, first_omega_0=cfg.first_omega_0, hidden_omega_0=cfg.hidden_omega_0
    ).cuda()

    optim = build_optimizer(img_siren, cfg.optimizer, cfg.learning_rate)
    tracked_omega_layer_idx, tracked_omega_layer = pick_tracked_omega_layer(img_siren, seed=cfg.omega_track_seed)

    snapshot_patch_set, snapshot_steps = sample_activation_snapshots(num_patches, cfg.total_steps, cfg.activation_snapshot_num_patches, cfg.activation_log_seed)
    activation_neuron_ids = sample_activation_neuron_ids(cfg.hidden_features, cfg.activation_num_neurons_to_log, cfg.activation_log_seed + 1)

    system_history = new_system_history()
    omega_history = []
    global_start = time.perf_counter()
    processes_patches = []

    for patch_idx, img_patch in enumerate(patches):
        patch = img_patch['patch']
        H, D, W = patch.shape

        dataloader = DataLoader(ImageFitting(patch, H, D, W), batch_size=1, pin_memory=True, num_workers=1)
        model_input, ground_truth = next(iter(dataloader))
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

        patch_noise = PatchNoiseModel(a=cfg.noise_model_a, b=cfg.noise_model_b, hu_bin_width=cfg.noise_model_hu_bin_width, overlap_ratio=cfg.noise_model_overlap_ratio, bins=cfg.noise_model_bins)
        patch_noise.setup_from_global(ground_truth, (H, D, W), img_patch['slice'], global_noise_reference, device=ground_truth.device)

        for step in range(cfg.total_steps):
            model_output, coords = img_siren(model_input)
            output_vol = model_output[0].view(H, D, W)
            gt_vol = ground_truth[0].view(H, D, W)

            gradient_model = torch.stack(torch.gradient(output_vol, spacing=1), dim=0)
            gradient_gt = torch.stack(torch.gradient(gt_vol, spacing=1), dim=0)
            grad_diff_mag = torch.sqrt(torch.sum((gradient_model - gradient_gt) ** 2, dim=0) + 1e-8)

            gradient_loss = grad_diff_mag.mean()
            mse_loss = ((model_output - ground_truth) ** 2).mean()
            tv_loss = tv_loss_3d(output_vol)

            zero_loss = torch.zeros((), dtype=model_output.dtype, device=model_output.device)
            noise_mse_loss = patch_noise.noise_mse_loss(model_output) if cfg.loss_noise_mse_weight != 0.0 else zero_loss
            noise_js_loss = patch_noise.noise_js_loss(model_output) if cfg.loss_noise_js_weight != 0.0 else zero_loss
            residual_noise_js_loss = patch_noise.residual_noise_js_loss(model_output, ground_truth)

            loss = (cfg.loss_mse_weight * mse_loss + cfg.loss_grad_weight * gradient_loss + cfg.loss_tv_weight * tv_loss + cfg.loss_noise_mse_weight * noise_mse_loss + cfg.loss_noise_js_weight * noise_js_loss + cfg.loss_residual_noise_js_weight * residual_noise_js_loss)

            global_step = patch_idx * cfg.total_steps + step

            if patch_idx in snapshot_patch_set and step == snapshot_steps[patch_idx]:
                attribution = LayerActivation(img_siren, img_siren.net[3]).attribute(model_input)
                log_all_neuron_activations_wandb(attribution, patch_idx, step, global_step, (H, D, W), activation_neuron_ids, cfg.activation_slice_axis, cfg.activation_slice_idx, cfg.hidden_features, cfg.activation_maps_dir)

            hw_metrics = get_hardware_metrics(cfg.gpu_device_index)
            append_system_history(system_history, hw_metrics, global_step)

            optim.zero_grad()
            loss.backward()
            optim.step()

            omega_val = get_sine_layers(img_siren)[tracked_omega_layer_idx].omega_0.item()
            omega_history.append((global_step, omega_val))

            if step % cfg.steps_til_summary == 0:
                wandb.log({"train/loss": loss.item()})
                print(f"[Patch {patch_idx+1:03d}] Step {step:04d}/{cfg.total_steps} | Loss: {loss.item():.6f}")

        processes_patches.append({"patch": model_output.squeeze(-1).reshape(H, D, W), "slice": img_patch['slice']})

    plot_omega_tracking(omega_history, tracked_omega_layer_idx, tracked_omega_layer.is_first, cfg.omega_plot_path)
    plot_system_resources(system_history, cfg.system_plot_path, cfg.gpu_device_index)

    torch.save({"model_state_dict": img_siren.state_dict(), "optimizer_state_dict": optim.state_dict(), "loss": loss.item(), "wandb_config": dict(cfg)}, "checkpoint.pth")
    wandb.save("checkpoint.pth")

    recon_img, _ = reconstruct_patches_monai(processes_patches, ori_shape)
    np.save("reconstructed_volume.npy", recon_img.detach().cpu().numpy())

    recon_np = recon_img.detach().cpu().numpy()
    img_np = img.detach().cpu().numpy()
    score, _ = ssim(recon_np, img_np, data_range=2, full=True, channel_axis=2)
    wandb.log({"metrics/ssim": score})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    compute_lpips_for_slice(recon_img[:, 115, :], img[:, 115, :], "Coronal [:,115,:]", loss_fn_alex, loss_fn_vgg, device)
    compute_lpips_for_slice(recon_img[:, :, 115], img[:, :, 115], "Sagittal [:,:,115]", loss_fn_alex, loss_fn_vgg, device)
    compute_lpips_for_slice(recon_img[100, :, :], img[100, :, :], "Axial [100,:,:]", loss_fn_alex, loss_fn_vgg, device)

    wandb.finish()

if __name__ == "__main__":
    main()