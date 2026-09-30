"""
Noise Component Extraction and Distribution Matching Framework
---------------------------------------------------------------

Purpose:
    - Extract noise from uniform medical regions
    - Model custom Poisson likelihood
    - Generate synthetic noise (Poisson + Uniform)
    - Apply log-rescaling transformation
    - Compare distributions using KL and JS divergence
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import factorial


# ============================================================
# OBJECT ORIENTED NOISE MODELING
# ============================================================

class CustomNoiseModel:
    def __init__(self, a=1.0, b=0.001, seed=None):
        self.a = a
        self.b = b
        self.seed = seed

        # Tracked internal states
        self.img_offset = None
        self.mean = None
        self.l_b = None
        self.u_b = None
        self.distribution_sorted = None

    def load_image(self, path):
        filename = os.path.basename(path)
        print(filename)
        img = np.load(path)
        return img, filename

    def normalize_image(self, img):
        self.img_offset = (img + 1001) * 100 / 6264
        print(self.img_offset.shape)
        return self.img_offset

    def build_custom_poisson_distribution(self, img_offset=None):
        if img_offset is None:
            img_offset = self.img_offset
            
        self.mean = np.mean(img_offset)
        self.l_b = np.min(img_offset)
        self.u_b = np.max(img_offset)

        print(f"mu: {self.mean: .2f}, u_b: {self.u_b: .2f}, l_b: {self.l_b: .2f}")

        distribution = {}
        for slice in img_offset:
            for row in slice:
                for bi in row:
                    probability_bi = ((self.mean ** bi) * np.exp(-self.mean)) / (factorial(bi))
                    distribution[bi] = probability_bi

        self.distribution_sorted = {k: distribution[k] for k in sorted(distribution.keys())}
        return self.distribution_sorted, self.mean, self.l_b, self.u_b

    def generate_custom_poisson_noise(self, distribution_sorted=None, shape=None):
        if distribution_sorted is None:
            distribution_sorted = self.distribution_sorted
            
        keys = np.array(list(distribution_sorted.keys()))
        values = np.array(list(distribution_sorted.values()))
        values_normalized = values / values.sum()

        if self.seed is not None:
            np.random.seed(self.seed)

        noise_flat = np.random.choice(keys, size=np.prod(shape), p=values_normalized)
        return noise_flat.reshape(shape)

    def generate_random_noise(self, shape, l_b=None, u_b=None):
        if l_b is None: l_b = self.l_b
        if u_b is None: u_b = self.u_b
            
        if self.seed is not None:
            np.random.seed(self.seed)

        return np.random.uniform(low=l_b, high=u_b, size=shape)

    def transform_and_rescale(self, combined_noise, u_b=None, l_b=None):
        if l_b is None: l_b = self.l_b
        if u_b is None: u_b = self.u_b
            
        #combined_log = u_b - (np.log(combined_noise / np.max(combined_noise)))

        #print(
        #   f"combined_log --> min: {np.min(combined_log):.2f}, "
        #   f"max: {np.max(combined_log):.2f}, "
        #   f"mean: {np.mean(combined_log):.2f}, "
        #   f"std: {np.std(combined_log):.2f}"
        #)

        I_I0 = (np.min(combined_noise) / combined_noise)
        print(
            f"I_I0 --> min: {np.min(I_I0)}, "
            f"max: {np.max(I_I0)}, "
            f"mean: {np.mean(I_I0)}, "
            f"std: {np.std(I_I0)}"
        )
        if l_b < 5: # Assumed that below -700HU in original image belongs to air/vacume region
            combined_log =  u_b + (np.log(I_I0))
        else:
            combined_log = - np.log(I_I0)
        # combined_log = np.log(1 + ((mean + combined_noise) / img_offset))

        print(
            f"combined_log --> min: {np.min(combined_log)}, "
            f"max: {np.max(combined_log)}, "
            f"mean: {np.mean(combined_log)}, "
            f"std: {np.std(combined_log)}"
        )

        # rescaled = (((u_b - combined_log) / (1)))
        rescaled = combined_log

        # rescaled = combined_noise
        rescaled_offset = (rescaled - np.min(rescaled))

        print(
            f"rescaled_offset --> min: {np.min(rescaled_offset)}, "
            f"max: {np.max(rescaled_offset)}, "
            f"mean: {np.mean(rescaled_offset)}, "
            f"std: {np.std(rescaled_offset)}"
        )

        #rescaled = combined_noise
        #rescaled_offset = (rescaled - np.min(rescaled))

        rescaled_offset_final = (
            rescaled_offset *
            ((u_b - l_b) / (np.max(rescaled_offset) - np.min(rescaled_offset)))
        ) + l_b

        return combined_log, rescaled, rescaled_offset_final

    def compute_pdf(self, data, bins=100, data_range=None, eps=1e-12):
        hist, bin_edges = np.histogram(data, bins=bins, range=data_range, density=False)
        pdf = hist.astype(np.float64)
        pdf /= np.sum(pdf)
        pdf = np.clip(pdf, eps, None)
        pdf /= np.sum(pdf)
        return pdf, bin_edges

    def kl_divergence(self, P, Q):
        return np.sum(P * np.log(P / Q))

    def js_divergence(self, P, Q):
        M = 0.5 * (P + Q)
        return 0.5 * self.kl_divergence(P, M) + 0.5 * self.kl_divergence(Q, M)

    def generate_combined_noise_metrics(self, img):
        noise_shape = np.shape(img)
        img_offset = self.normalize_image(img)
        dist_sorted, _, l_b, u_b = self.build_custom_poisson_distribution(img_offset)

        poisson_noise = self.generate_custom_poisson_noise(dist_sorted, shape=noise_shape)
        uniform_noise = self.generate_random_noise(shape=noise_shape, l_b=l_b, u_b=u_b)

        combined_noise = (self.a * poisson_noise) + (self.b * uniform_noise)
        _, _, rescaled_offset_final = self.transform_and_rescale(combined_noise, u_b, l_b)
        
        return poisson_noise, uniform_noise, combined_noise, rescaled_offset_final


# ============================================================
# VISUALIZATION
# ============================================================

def plot_noise_statistics(keys, values,
                          poisson_noise,
                          uniform_noise,
                          rescaled_offset_final,
                          l_b, u_b):

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].bar(keys, values, width=1, color='steelblue', edgecolor='none')
    axes[0].set_title("Custom Poisson Likelihood")
    axes[0].set_xlabel("Pixel Intensity (k)")
    axes[0].set_ylabel("Poisson Likelihood P(k|µ)")

    axes[1].hist(poisson_noise.flatten(), bins=100, color='steelblue', edgecolor='none')
    axes[1].set_title("Custom Poisson Noise")

    axes[2].hist(uniform_noise.flatten(), bins=100, color='tomato', edgecolor='none')
    axes[2].set_title(f"Uniform Random Noise [{l_b/10:.0f}, {u_b/10:.0f}]")

    axes[3].hist(rescaled_offset_final.flatten(), bins=100,
                 color='mediumseagreen', edgecolor='none')
    axes[3].set_title("Combined Noise model")

    plt.tight_layout()
    plt.show()


def plot_pdf_comparison(P, Q, bin_edges, filename):

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, P, label="Original PDF")
    plt.plot(bin_centers, Q, label="Rescaled PDF")

    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title(f"PDF Comparison (100 bins) {filename}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Parameters
    path = r"C:\Users\thiks\Downloads\patch_100.npy"
    noise_model = CustomNoiseModel(a=1, b=0.001, seed=25)

    img, filename = noise_model.load_image(path)
    print(f"noise_shape: {np.shape(img)}")
    print(f"Image max: {np.max(img):.2f}, Image min: {np.min(img):.2f}")

    img_offset = noise_model.normalize_image(img)
    print(f"Image offset min: {np.min(img_offset):.2f}, Image offset max: {np.max(img_offset):.2f}")

    distribution_sorted, mean, l_b, u_b = noise_model.build_custom_poisson_distribution(img_offset)

    keys = list(distribution_sorted.keys())
    values = list(distribution_sorted.values())

    # Full sequence wrapper
    poisson_noise, uniform_noise, combined_noise, rescaled_offset_final = \
        noise_model.generate_combined_noise_metrics(img)

    print(f"Poisson noise  -> min: {np.min(poisson_noise):.2f}, max: {np.max(poisson_noise):.2f}, mean: {np.mean(poisson_noise):.2f}")
    print(f"Uniform noise  -> min: {np.min(uniform_noise):.2f}, max: {np.max(uniform_noise):.2f}, mean: {np.mean(uniform_noise):.2f}")
    print(f"Combined noise -> min: {np.min(combined_noise):.2f}, max: {np.max(combined_noise):.2f}, mean: {np.mean(combined_noise):.2f}")

    # Plot noise distributions
    plot_noise_statistics(keys, values, poisson_noise, uniform_noise, rescaled_offset_final, l_b, u_b)

    # Divergence analysis
    img_flat = noise_model.img_offset.flatten()
    rescaled_flat = rescaled_offset_final.flatten()

    common_min = min(img_flat.min(), rescaled_flat.min())
    common_max = max(img_flat.max(), rescaled_flat.max())

    P, bin_edges = noise_model.compute_pdf(img_flat, bins=100, data_range=(common_min, common_max))
    Q, _ = noise_model.compute_pdf(rescaled_flat, bins=100, data_range=(common_min, common_max))

    kl_PQ = noise_model.kl_divergence(P, Q)
    kl_QP = noise_model.kl_divergence(Q, P)
    js = noise_model.js_divergence(P, Q)

    print("KL(img || rescaled): %.6f" % kl_PQ)
    print("KL(rescaled || img): %.6f" % kl_QP)
    print("JS divergence: %.6f" % js)

    plot_pdf_comparison(P, Q, bin_edges, filename)

    # Note: GlobalNoiseReference should only be imported here if explicitly required 
    # to avoid circular dependency since losses imports noise_model.


if __name__ == "__main__":
    main()