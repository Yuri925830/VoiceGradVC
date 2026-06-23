import torch
import torch.nn as nn
import math


class VoiceGradDiffusion(nn.Module):
    def __init__(self, n_levels=20, offset=0.008):
        super().__init__()
        self.n_levels = n_levels
        self.offset = offset

        # =========================================================
        # 1. Cosine Noise Schedule (Strictly following paper V-E equation 21)
        #
        # alpha_bar_l = f(l) / f(0)
        # f(l) = cos(((l / L) + eta) / (1 + eta) * pi / 2)^2
        #
        # beta_l = 1 - alpha_bar_l / alpha_bar_{l-1}
        # beta_l clipped to <= 0.999
        # =========================================================
        steps = torch.arange(n_levels + 1, dtype=torch.float64) / n_levels

        f = torch.cos(
            ((steps + offset) / (1.0 + offset)) * (math.pi / 2.0)
        ) ** 2

        alpha_bar = f / f[0]  # shape: [L+1], alpha_bar[0] = 1

        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])  # shape: [L]
        betas = torch.clamp(betas, min=0.0, max=0.999)

        betas = betas.float()
        alphas = (1.0 - betas).float()

        # Note: 
        # Re-calculating alphas_cumprod using clipped betas ensures 
        # consistency between training and sampling.
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()

        # =========================================================
        # 2. Precompute common coefficients
        # =========================================================
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("recip_sqrt_alphas", 1.0 / torch.sqrt(alphas))

        # Coefficients for Algorithm 4:
        # (1 - alpha_l) / sqrt(1 - alpha_bar_l)
        self.register_buffer(
            "remove_noise_coeff",
            betas / torch.sqrt(1.0 - alphas_cumprod)
        )

        # In the paper nu_l^2 = beta_l, so nu_l = sqrt(beta_l)
        self.register_buffer("sigma", torch.sqrt(betas))

    def get_index(self, tensor, t, shape):
        """
        Extract the value at time t for each sample in the batch from a 
        schedule of shape [L], then reshape for broadcasting.
        """
        batch_size = t.shape[0]
        out = tensor.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion (used during training).
        Formula:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = self.get_index(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alpha_bar_t = self.get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    @torch.no_grad()
    def sample(self, model, x_source, speaker_id, bnf, start_level=11):
        """
        Sampling using DPM-based VoiceGrad (Algorithm 4 from the paper)

        Args:
            model: Trained VoiceGrad model
            x_source: Source speech mel (normalized) [B, 80, T]
            speaker_id: Target speaker ID [B]
            bnf: BNF features [B, 144, T]
            start_level: L' from the paper, default 11

        Returns:
            x: Converted mel features
        """
        if start_level < 1 or start_level > self.n_levels:
            raise ValueError(
                f"start_level must be in [1, {self.n_levels}], but got {start_level}"
            )

        # Approach: Start reverse diffusion directly from source mel
        x = x_source.clone()
        batch_size = x.shape[0]
        device = x.device

        # Loop from l = L' ... 1
        # Explicitly maintain 1-based semantics from the paper, then map to 0-based index
        for l in range(start_level, 0, -1):
            t = torch.full((batch_size,), l - 1, device=device, dtype=torch.long)

            # epsilon_theta(x, l, k)
            predicted_noise = model(x, t, speaker_id, bnf)

            recip_sqrt_alpha = self.get_index(self.recip_sqrt_alphas, t, x.shape)
            noise_coeff = self.get_index(self.remove_noise_coeff, t, x.shape)
            sigma = self.get_index(self.sigma, t, x.shape)

            # Algorithm 4:
            # x <- 1/sqrt(alpha_l) * (x - (1-alpha_l)/sqrt(1-alpha_bar_l) * eps_theta) + nu_l * z
            mean = recip_sqrt_alpha * (x - noise_coeff * predicted_noise)

            # Draw z ~ N(0, I) at each step
            z = torch.randn_like(x)

            x = mean + sigma * z

        return x


if __name__ == "__main__":
    # =========================
    # Minimal smoke test
    # =========================
    diffusion = VoiceGradDiffusion(n_levels=20, offset=0.008)

    batch_size = 2
    x0 = torch.randn(batch_size, 80, 128)
    t = torch.randint(0, 20, (batch_size,))
    xt = diffusion.q_sample(x0, t)

    print("x0 shape:", x0.shape)
    print("xt shape:", xt.shape)

    # Check schedule validity
    print("betas shape:", diffusion.betas.shape)
    print("alphas_cumprod shape:", diffusion.alphas_cumprod.shape)
    print("beta min/max:", diffusion.betas.min().item(), diffusion.betas.max().item())
