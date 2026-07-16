import random
import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class SineLayer(nn.Module):
    # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 
    # for discussion of omega_0.
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=10):
        super().__init__()
        # TRAINABLE OMEGA
        self.omega_0 = nn.Parameter(torch.tensor(float(omega_0)))
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=False, first_omega_0=10, hidden_omega_0=10.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # allows to take derivative w.r.t. input
        coords = coords.clone().detach().requires_grad_(True) 
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
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
    return {f"omega_initial/layer_{i}": layer.omega_0.detach().item() for i, layer in enumerate(layers)}

omega_initial_values = collect_initial_omega_values(img_siren)
wandb.config.update(omega_initial_values, allow_val_change=True)

log_dict = {"train/loss": loss.item()}
log_dict.update(collect_omega_metrics(img_siren, tracked_omega_layer_idx))
wandb.log(log_dict, step=global_step)