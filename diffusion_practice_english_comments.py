# Import the main PyTorch library; torch is used to create tensors, generate random noise, and perform all numerical computations.
import torch
# Import torch.nn as nn; VoiceGradDiffusion inherits nn.Module so its buffers can move to the GPU together with the model.
import torch.nn as nn
# Import Python's built-in math library; this file mainly needs math.pi, the constant pi.
import math


# Define the VoiceGradDiffusion class; this class controls how noise is added and how it is removed step by step.
class VoiceGradDiffusion(nn.Module):
    # Define the initialization method; it runs automatically when the object is created and prepares all diffusion coefficient tables in advance.
    def __init__(self, n_levels=20, offset=0.008):
        # Call the parent nn.Module initializer; without this line, PyTorch cannot correctly manage buffers, device movement, and other module features.
        super().__init__()
        # Save the total number of diffusion levels; the DPM version in the paper uses L=20, so the default value is 20.
        self.n_levels = n_levels
        # Save the small offset eta used by the cosine schedule; the paper sets it to 0.008 to prevent the earliest beta values from becoming too small.
        self.offset = offset

        # Start building the cosine noise schedule from Equation 21 in the paper; this schedule decides how much noise each diffusion step adds.
        # torch.arange(n_levels + 1) creates 0,1,2,...,20; we need 21 points because the schedule must also include the starting point l=0.
        steps = torch.arange(n_levels + 1, dtype=torch.float64) / n_levels

        # torch.cos(...) computes cosine; this line directly implements f(l)=cos(((l/L)+eta)/(1+eta)*pi/2)^2 from the paper.
        f = torch.cos(
            # steps already represents l/L; adding offset, dividing by 1+offset, and multiplying by pi/2 forms the complete cosine input in the paper.
            ((steps + offset) / (1.0 + offset)) * (math.pi / 2.0)
        # ** 2 means square; the paper requires the whole cosine result to be squared.
        ) ** 2

        # Divide f(l) by f(0) to obtain the theoretical alpha_bar schedule; f[0] selects the first value so alpha_bar starts at 1.
        alpha_bar = f / f[0]

        # alpha_bar[1:] selects elements from index 1 to the end, while alpha_bar[:-1] selects from the beginning to the second-to-last element.
        # These two slices represent adjacent time steps, allowing beta_l to be computed as 1-alpha_bar_l/alpha_bar_{l-1}.
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        # torch.clamp limits beta to the range 0 to 0.999; the paper prevents beta from getting too close to 1 because that would cause numerical instability.
        betas = torch.clamp(betas, min=0.0, max=0.999)

        # .float() converts float64 to the float32 type normally used by the model; this saves memory and matches the dtype of mel features and model parameters.
        betas = betas.float()
        # Compute alpha at each step from alpha=1-beta; the outer .float() again ensures the result is float32.
        alphas = (1.0 - betas).float()

        # torch.cumprod(..., dim=0) performs cumulative multiplication along dimension 0, producing alpha_1, alpha_1*alpha_2, and so on.
        # This result is the alpha_bar table actually used for training and sampling; recomputing it from the clipped betas keeps both processes fully consistent.
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()

        # register_buffer registers a tensor as a fixed internal value of the module; the optimizer does not update it, but model.to('cuda') moves it to the GPU automatically.
        self.register_buffer("betas", betas)
        # Store alpha for every time step; the reverse diffusion formula uses these values directly.
        self.register_buffer("alphas", alphas)
        # Store alpha_bar for every time step, which is the cumulative product of alpha.
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Precompute sqrt(alpha); doing this once avoids repeated square-root calculations in every batch and sampling step.
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        # Precompute sqrt(alpha_bar); the forward diffusion function q_sample uses it directly.
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        # The first argument of register_buffer is the saved name, and the second argument is the tensor to register.
        self.register_buffer(
            # This name represents sqrt(1-alpha_bar); later it can be accessed as self.sqrt_one_minus_alphas_cumprod.
            "sqrt_one_minus_alphas_cumprod",
            # torch.sqrt computes a square root; this coefficient controls how much random noise appears in forward diffusion.
            torch.sqrt(1.0 - alphas_cumprod)
        # This closing parenthesis ends the current register_buffer call.
        )
        # Precompute 1/sqrt(alpha); this is the outer multiplier in Algorithm 4 of the paper.
        self.register_buffer("recip_sqrt_alphas", 1.0 / torch.sqrt(alphas))

        # Register the noise-removal coefficient (1-alpha_l)/sqrt(1-alpha_bar_l) used by the reverse diffusion formula.
        self.register_buffer(
            # Name this coefficient table remove_noise_coeff because it scales the predicted noise that will be subtracted.
            "remove_noise_coeff",
            # Since 1-alpha equals beta, the code directly uses beta / sqrt(1-alpha_bar).
            betas / torch.sqrt(1.0 - alphas_cumprod)
        # This closing parenthesis ends the register_buffer call.
        )

        # The paper sets nu_l^2=beta_l, so nu_l=sqrt(beta_l); sigma controls the amount of randomness added back at each reverse diffusion step.
        self.register_buffer("sigma", torch.sqrt(betas))

    # Define the helper function get_index; it takes one coefficient from a length-L table for each sample according to that sample's own time step.
    def get_index(self, tensor, t, shape):
        # t.shape[0] reads the size of dimension 0; because t has shape [B], this value is the batch size.
        batch_size = t.shape[0]
        # tensor.gather(0, t) selects values along dimension 0 using the indices stored in t, so every sample receives the coefficient for its own time step.
        out = tensor.gather(0, t)
        # reshape changes [B] into [B,1,1], which allows automatic broadcasting with mel tensors shaped [B,80,T].
        # *((1,) * (len(shape) - 1)) uses Python unpacking syntax to add the required number of singleton dimensions automatically.
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    # Define q_sample; this function performs forward diffusion during training, turning clean mel x_0 into noisy mel x_t.
    def q_sample(self, x_start, t, noise=None):
        # If the caller does not provide noise, let the function generate standard Gaussian noise by itself.
        if noise is None:
            # torch.randn_like(x_start) creates N(0,1) noise with exactly the same shape, device, and dtype as x_start.
            noise = torch.randn_like(x_start)

        # For every sample in the batch, take the sqrt(alpha_bar) coefficient that corresponds to its own time step t.
        sqrt_alpha_bar_t = self.get_index(
            # Pass the complete sqrt(alpha_bar) coefficient table.
            self.sqrt_alphas_cumprod,
            # Pass the time-step index t for each sample; its shape is [B].
            t,
            # Pass x_start.shape so get_index knows how many dimensions the output needs for broadcasting.
            x_start.shape
        # This closing parenthesis ends the get_index call.
        )
        # In the same way, take sqrt(1-alpha_bar_t), which determines how much random noise appears in x_t.
        sqrt_one_minus_alpha_bar_t = self.get_index(
            # Pass the complete sqrt(1-alpha_bar) coefficient table.
            self.sqrt_one_minus_alphas_cumprod,
            # Use the same time-step indices t for this batch.
            t,
            # Use the shape of x_start to determine the broadcasting dimensions.
            x_start.shape
        # This closing parenthesis ends the get_index call.
        )

        # Directly implement the paper equation x_t=sqrt(alpha_bar_t)*x_0+sqrt(1-alpha_bar_t)*epsilon.
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    # @torch.no_grad() is decorator syntax; it disables gradient recording for the whole sample function because inference only needs outputs, not backpropagation.
    @torch.no_grad()
    # Define sample; this function denoises the source mel step by step from start_level according to Algorithm 4 in the paper.
    def sample(self, model, x_source, speaker_id, bnf, start_level=11):
        # First check whether start_level is inside the valid range 1 to L; an invalid value raises an error immediately instead of accessing a nonexistent step.
        if start_level < 1 or start_level > self.n_levels:
            # raise stops the program and throws an exception; ValueError means that an input argument has an invalid value.
            raise ValueError(
                # An f-string inserts variable values directly into a string, so the error message can show both the valid range and the actual input.
                f"start_level must be in [1, {self.n_levels}], but got {start_level}"
            # This closing parenthesis ends the ValueError call.
            )

        # clone() makes a copy of the source mel, so later updates to x do not modify the original x_source passed in from outside.
        x = x_source.clone()
        # x.shape[0] reads the batch dimension; we need it when creating a time-step tensor for every sample in the batch.
        batch_size = x.shape[0]
        # x.device returns the current device of x, such as cpu or cuda; newly created tensors must be placed on the same device.
        device = x.device

        # range(start_level,0,-1) counts backward from start_level to 1, for example 11,10,...,1.
        for l in range(start_level, 0, -1):
            # torch.full creates a tensor of shape [B] filled with l-1; subtracting 1 converts the paper's 1-based index into Python's 0-based index.
            t = torch.full((batch_size,), l - 1, device=device, dtype=torch.long)

            # Ask the model to predict the noise epsilon_theta in the current x while also providing the time step, target speaker, and source-speech BNF.
            predicted_noise = model(x, t, speaker_id, bnf)

            # Take 1/sqrt(alpha_l) for the current step and reshape it so it can broadcast with x.
            recip_sqrt_alpha = self.get_index(self.recip_sqrt_alphas, t, x.shape)
            # Take the current noise-removal coefficient beta_l/sqrt(1-alpha_bar_l).
            noise_coeff = self.get_index(self.remove_noise_coeff, t, x.shape)
            # Take sigma=sqrt(beta_l) for the current step; it controls the strength of the random noise added back.
            sigma = self.get_index(self.sigma, t, x.shape)

            # Compute the mean part of Algorithm 4, excluding random z; this uses the model prediction to move the current x toward a cleaner state.
            mean = recip_sqrt_alpha * (x - noise_coeff * predicted_noise)

            # Generate standard Gaussian noise z with exactly the same shape as x; Algorithm 4 requires drawing z~N(0,I) at every step.
            z = torch.randn_like(x)

            # Add sigma*z to mean to obtain the next x, completing one reverse diffusion update from x_l to x_{l-1}.
            x = mean + sigma * z

        # After all reverse steps are finished, return the converted mel; its shape is still [B,80,T], the same as x_source.
        return x


# The self-test code below runs only when this file is executed directly; importing the file from another module does not run the test.
if __name__ == "__main__":
    # Create a diffusion object with 20 levels and offset=0.008 to check whether the formulas and tensor shapes work correctly.
    diffusion = VoiceGradDiffusion(n_levels=20, offset=0.008)

    # Set the self-test batch size to 2, meaning that two speech samples are simulated at the same time.
    batch_size = 2
    # Create a random tensor with shape [2,80,128] and treat it as two normalized clean mel spectrograms.
    x0 = torch.randn(batch_size, 80, 128)
    # torch.randint(0,20,(2,)) randomly chooses one diffusion step from 0 to 19 for each of the two samples.
    t = torch.randint(0, 20, (batch_size,))
    # Call q_sample to add noise to x0 and obtain xt at the selected time steps.
    xt = diffusion.q_sample(x0, t)

    # Print the shape of x0 to confirm that the clean input before diffusion is [2,80,128].
    print("x0 shape:", x0.shape)
    # Print the shape of xt to confirm that adding noise does not change the mel shape.
    print("xt shape:", xt.shape)

    # Print the shape of the beta table; when n_levels=20, there should be 20 beta values.
    print("betas shape:", diffusion.betas.shape)
    # Print the shape of the alpha_bar table; it should also contain 20 values, one for each diffusion level.
    print("alphas_cumprod shape:", diffusion.alphas_cumprod.shape)
    # .min()/.max() find the smallest and largest beta, and .item() converts a one-element tensor into a normal Python number for printing.
    print("beta min/max:", diffusion.betas.min().item(), diffusion.betas.max().item())
