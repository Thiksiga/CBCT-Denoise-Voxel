import torch
import numpy as np
import skimage
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from monai.data.utils import dense_patch_slices


def extract_patches_monai(volume: torch.Tensor, patch_size, stride):
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


def img_normalisation(img: torch.Tensor):
    img = img.float()
    min_val = -1000
    max_val = 5264
    img = (img - min_val) / (max_val - min_val)
    img = img * 2 - 1
    return img


def create_3d_grid(H, D, W, device='cpu'):
    z = torch.linspace(-1, 1, steps=D, device=device)
    y = torch.linspace(-1, 1, steps=H, device=device)
    x = torch.linspace(-1, 1, steps=W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([zz, yy, xx], dim=-1)
    return grid


def create_3d_coords(H, D, W, device='cpu'):
    grid = create_3d_grid(H, D, W, device)
    coords = grid.view(-1, 3)
    return coords


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    print(f"img 1: {img.size}")
    img = img.resize((sidelength,) * 2)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    print(f"img 2: {img.shape}")
    return img


class ImageFitting(Dataset):
    def __init__(self, img, H, D, W):
        super().__init__()
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
        img = img
        self.pixels = img.view(-1, 1)


class SIRENPatchDataset(Dataset):
    def __init__(self, patches, normalize=True):
        self.patches = patches
        self.normalize = normalize
        sample_patch = patches[0]["patch"]
        self.H, self.D, self.W = sample_patch.shape
        self.coords = create_3d_coords(self.H, self.D, self.W)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]["patch"].float()
        if self.normalize:
            patch = (patch + 1000) / 6264
            patch = patch * 2 - 1
        pixels = patch.reshape(-1, 1)
        return self.coords, pixels