import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import factorial


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


def tv_loss_3d(volume: torch.Tensor) -> torch.Tensor:
    dz = torch.abs(volume[1:, :, :] - volume[:-1, :, :]).mean()
    dy = torch.abs(volume[:, 1:, :] - volume[:, :-1, :]).mean()
    dx = torch.abs(volume[:, :, 1:] - volume[:, :, :-1]).mean()
    return dx + dy + dz


def _tensor_to_hu(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1.0) * 0.5 * 6264.0 - 1000.0


def _normalize_patch_offset(hu_patch: np.ndarray) -> np.ndarray:
    return (hu_patch + 1001.0) * 100.0 / 6264.0


def _tensor_to_patch_offset(tensor, shift_min, shift_max):
    hu = _tensor_to_hu(tensor)
    return (hu + 1001.0) * 100.0 / 6264.0


def _build_overlapping_range_poisson_distribution(
    img_offset, hu_bin_width=200, overlap_ratio=0.05, original_range=6264.0, normalized_range=100.0,
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
        flat = img_offset.reshape(-1).astype(np.float64)
        unique_values, counts = np.unique(flat, return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()
        return dict(zip(unique_values.tolist(), probabilities.tolist())), l_b, u_b

    distribution_sorted = {k: merged_distribution[k] for k in sorted(merged_distribution)}
    total_probability = sum(distribution_sorted.values())
    if not np.isfinite(total_probability) or total_probability <= 0:
        flat = img_offset.reshape(-1).astype(np.float64)
        unique_values, counts = np.unique(flat, return_counts=True)
        probabilities = counts.astype(np.float64) / counts.sum()
        return dict(zip(unique_values.tolist(), probabilities.tolist())), l_b, u_b

    distribution_sorted = {k: v / total_probability for k, v in distribution_sorted.items()}
    return distribution_sorted, l_b, u_b


def _generate_custom_poisson_noise(distribution_sorted, shape, seed=42):
    rng = np.random.default_rng(seed)
    keys = np.array(list(distribution_sorted.keys()), dtype=np.float64)
    values = np.array(list(distribution_sorted.values()), dtype=np.float64)
    values_normalized = values / values.sum()
    noise_flat = rng.choice(keys, size=np.prod(shape), p=values_normalized)
    return noise_flat.reshape(shape)


def _generate_random_noise(shape, l_b, u_b, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=l_b, high=u_b, size=shape)


def _transform_and_rescale_noise(combined_noise, u_b, l_b):
    combined_noise = np.asarray(combined_noise, dtype=np.float64)
    max_val = np.max(combined_noise)
    combined_log = u_b - np.log(combined_noise / max_val + 1e-12)
    rescaled = u_b - combined_log
    rescaled_offset = rescaled - np.min(rescaled)
    denominator = np.max(rescaled_offset) - np.min(rescaled_offset)
    if denominator < 1e-12:
        return np.full_like(rescaled_offset, l_b)
    return rescaled_offset * ((u_b - l_b) / denominator) + l_b


def _compute_pdf_np(data, bins=100, data_range=None, eps=1e-12):
    hist, bin_edges = np.histogram(data, bins=bins, range=data_range, density=False)
    pdf = hist.astype(np.float64)
    pdf /= np.sum(pdf) + eps
    pdf = np.clip(pdf, eps, None)
    pdf /= np.sum(pdf)
    return pdf, bin_edges


def _js_divergence_np(P, Q, eps=1e-12):
    P = np.clip(np.asarray(P, dtype=np.float64), eps, None)
    Q = np.clip(np.asarray(Q, dtype=np.float64), eps, None)
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


def build_global_noise_reference(
    normalized_volume: torch.Tensor, a=1.0, b=0.01, hu_bin_width=200, overlap_ratio=0.05, bins=100, seed=42,
):
    hu_volume = _tensor_to_hu(normalized_volume.detach()).cpu().numpy()
    original_range = float(hu_volume.max() - hu_volume.min())
    if original_range < 1e-8:
        original_range = 6264.0

    img_offset = _normalize_patch_offset(hu_volume)
    shifted = hu_volume + 1001.0
    shift_min = float(np.min(shifted))
    shift_max = float(np.max(shifted))

    distribution_sorted, l_b, u_b = _build_overlapping_range_poisson_distribution(
        img_offset, hu_bin_width=hu_bin_width, overlap_ratio=overlap_ratio, original_range=original_range,
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


class PatchNoiseModel:
    def __init__(self, a=1.0, b=0.01, hu_bin_width=200, overlap_ratio=0.05, bins=100):
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
            img_offset, hu_bin_width=self.hu_bin_width, overlap_ratio=self.overlap_ratio, original_range=original_range,
        )

        shape = tuple(patch_shape)
        poisson_noise = _generate_custom_poisson_noise(distribution_sorted, shape, seed=seed)
        uniform_noise = _generate_random_noise(shape, l_b, u_b, seed=seed + 1)
        combined_noise = self.a * poisson_noise + self.b * uniform_noise
        rescaled_noise = _transform_and_rescale_noise(combined_noise, u_b, l_b)

        self.reference_noise_offset = torch.tensor(rescaled_noise, dtype=torch.float32, device=device)

        ref_flat = rescaled_noise.reshape(-1)
        common_min = float(ref_flat.min())
        common_max = float(ref_flat.max())
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0

        pdf_np, edges_np = _compute_pdf_np(ref_flat, bins=self.bins, data_range=(common_min, common_max))
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
        self.reference_residual_pdf = torch.tensor(residual_pdf_np, dtype=torch.float32, device=device)
        self.residual_bin_edges = torch.tensor(residual_edges_np, dtype=torch.float32, device=device)
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
        hu_patch = _tensor_to_hu(ground_truth.detach()).cpu().numpy().reshape(patch_shape)
        self.shift_min = float(global_reference["shift_min"])
        self.shift_max = float(global_reference["shift_max"])

        rescaled_noise = global_reference["noise_offset"][patch_slice].detach().cpu().numpy().reshape(patch_shape)
        self.reference_noise_offset = torch.tensor(rescaled_noise, dtype=torch.float32, device=device)

        ref_flat = rescaled_noise.reshape(-1)
        common_min = float(ref_flat.min())
        common_max = float(ref_flat.max())
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0

        pdf_np, edges_np = _compute_pdf_np(ref_flat, bins=self.bins, data_range=(common_min, common_max))
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
        self.reference_residual_pdf = torch.tensor(residual_pdf_np, dtype=torch.float32, device=device)
        self.residual_bin_edges = torch.tensor(residual_edges_np, dtype=torch.float32, device=device)
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
        model_offset = _tensor_to_patch_offset(model_output, self.shift_min, self.shift_max).reshape(-1)
        target = self.reference_noise_offset.reshape(-1)
        return F.mse_loss(model_offset, target)

    def noise_js_loss(self, model_output):
        model_offset = _tensor_to_patch_offset(model_output, self.shift_min, self.shift_max).reshape(-1)
        P = _soft_histogram_torch(model_offset, self.bin_edges)
        Q = self.reference_pdf / self.reference_pdf.sum()
        return _js_divergence_torch(P, Q)

    def residual_noise_js_loss(self, model_output, ground_truth):
        model_offset = _tensor_to_patch_offset(model_output, self.shift_min, self.shift_max)
        gt_offset = _tensor_to_patch_offset(ground_truth, self.shift_min, self.shift_max)
        residual_offset = (gt_offset - model_offset).reshape(-1)
        residual_offset = residual_offset - residual_offset.mean()
        P = _soft_histogram_torch(residual_offset, self.residual_bin_edges)
        Q = self.reference_residual_pdf / self.reference_residual_pdf.sum()
        return _js_divergence_torch(P, Q)

    def patch_js_numpy(self, model_output):
        model_offset = _tensor_to_patch_offset(model_output.detach(), self.shift_min, self.shift_max).cpu().numpy().reshape(-1)
        ref_flat = self.reference_noise_offset.detach().cpu().numpy().reshape(-1)
        common_min = float(min(model_offset.min(), ref_flat.min()))
        common_max = float(max(model_offset.max(), ref_flat.max()))
        if common_max - common_min < 1e-8:
            common_max = common_min + 1.0
        P, _ = _compute_pdf_np(model_offset, bins=self.bins, data_range=(common_min, common_max))
        Q, _ = _compute_pdf_np(ref_flat, bins=self.bins, data_range=(common_min, common_max))
        return _js_divergence_np(P, Q)

    def residual_patch_js_numpy(self, model_output, ground_truth):
        model_offset = _tensor_to_patch_offset(model_output.detach(), self.shift_min, self.shift_max).cpu().numpy().reshape(-1)
        gt_offset = _tensor_to_patch_offset(ground_truth.detach(), self.shift_min, self.shift_max).cpu().numpy().reshape(-1)
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