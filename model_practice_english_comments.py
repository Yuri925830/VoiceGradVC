import torch # `torch` is used to create tensors, generate random test inputs, and check model output shapes.
import torch.nn as nn # `torch.nn` provides the neural-network layers used to build the model.
import torch.nn.functional as F # `torch.nn.functional` provides function-style operations such as padding and GLU.

# 1. Define one reusable convolutional block used throughout the VoiceGrad network.
class VoiceGradBlock(nn.Module):
    def __init__(
        self,
        in_ch, # Number of input channels received by this block.
        out_ch, # Number of output channels that this block should produce after GLU.
        k, # Convolution kernel size (`kernel_size`), which controls how many nearby time positions are viewed at once.
        s, # Convolution step size, also called `stride`; a value of 2 performs temporal downsampling or upsampling.
        cond_dim, # Dimension of one condition embedding vector, such as the noise-level or speaker embedding.
        bnf_dim, # Original BNF channel dimension before projection.
        bnf_out, # Number of BNF channels after projection.
        bnf_stride=1, # Stride used by the BNF projection convolution so its time length matches the current U-Net stage.
        transpose=False # By default, this block uses a normal convolution rather than a transposed convolution.
    ):
        super().__init__() # Initialize the `nn.Module` parent class so PyTorch can register and manage all submodules and parameters.
        self.transpose = transpose # Save `transpose` as an object attribute so this block remembers which convolution type it uses.
        self.bnf_stride = bnf_stride # Save the BNF stride for later length checks and error messages.
        # This strided convolution is used only for BNF; it reduces the BNF time resolution to approximately the current block input length.
        self.bnf_proj = nn.utils.weight_norm( # Define the BNF projection layer that converts the original BNF into the compact conditioning feature used by this block.
            nn.Conv1d( # Define a one-dimensional convolution because the feature sequence is organized along one time axis.
                bnf_dim, # BNF input channel count, which is 144 in this reproduction.
                bnf_out, # BNF output channel count, reduced to 32 as the compact conditioning representation.
                kernel_size=1, # Use a 1x1 one-dimensional convolution to change channels without mixing neighboring time positions.
                stride=bnf_stride, # Use the stride to control the BNF time length for the current U-Net resolution.
                padding=0 # Do not add padding because a kernel size of 1 does not shrink the sequence by itself.
            )
        )

        total_in_ch = in_ch + (cond_dim * 2) + bnf_out # Compute the true convolution input channels: x channels + noise-condition channels + speaker-condition channels + BNF channels.
        glu_out_ch = out_ch * 2 # Double the convolution output channels because GLU later splits them into a content half and a gate half.

        if transpose: # Check whether this block should use a transposed convolution for temporal upsampling.
            # Decoder upsampling blocks use transposed convolution.
            padding = (k - s) // 2 # Compute padding for the transposed convolution so the output length is close to the intended doubled length.
            self.conv = nn.utils.weight_norm( # Define the main convolution of this block and wrap it with weight normalization.
                nn.ConvTranspose1d( # Define a one-dimensional transposed convolution for temporal upsampling.
                    total_in_ch, # Input channel count of the transposed convolution.
                    glu_out_ch, # Output channel count before GLU; it is doubled because GLU will split it in half.
                    kernel_size=k, # Convolution kernel size.
                    stride=s, # Convolution stride.
                    padding=padding # Use the padding calculated above so the upsampled output length stays close to the target length.
                )
            )
        else:
            # Encoder blocks and length-preserving/downsampling blocks use a normal convolution rather than a transposed convolution.
            padding = (k - 1) // 2 # Compute padding for the normal convolution so stride-1 layers preserve time length.
            self.conv = nn.utils.weight_norm( # Define the normal convolution and apply weight normalization.
                nn.Conv1d( # Define a one-dimensional convolution layer.
                    total_in_ch, # Number of input channels received by this block.
                    glu_out_ch, # Number of output channels that this block should produce after GLU.
                    kernel_size=k, # Convolution kernel size.
                    stride=s, # Convolution stride.
                    padding=padding
                )
            )

    # 2. Match the BNF feature time length to the current main-branch feature x.
    def _match_time_length(self, x, target_len):
        if x.shape[-1] < target_len: # If the current sequence is shorter than the required target length.
            diff = target_len - x.shape[-1] # Calculate how many time frames are missing.
            x = F.pad(x, (0, diff)) # Pad `diff` zeros on the right side of the time axis.
        elif x.shape[-1] > target_len: # If the current sequence is longer than the target length.
            x = x[..., :target_len] # Keep only the first `target_len` frames and crop the extra frames on the right.
        return x # Return x after its time length has been matched to the target length.

    # 3. Define the real data flow of this block, meaning how the inputs are processed during the forward pass.
    def forward(self, x, cond, bnf):
        if bnf is None: # If no BNF conditioning feature was provided.
            raise ValueError("VoiceGradBlock requires bnf input, but got None.") # Raise an error immediately because this reproduction requires BNF conditioning.

        T = x.shape[-1] # Read the current time length T from the main-branch feature x.

        # Expand the noise and speaker condition embeddings along the time axis to the current length T.
        cond_expanded = cond.expand(-1, -1, T) # Shape change: [B, 2*cond_dim, 1] -> [B, 2*cond_dim, T].

        # Send the original BNF through this block's own BNF projection layer.
        bnf_feat = self.bnf_proj(bnf) # Shape change: [B, 144, original T] -> [B, 32, a time length close to the current block T].

        # Normally the projected BNF length should already be almost identical to x, with at most a one-frame difference.
        if abs(bnf_feat.shape[-1] - T) > 1: # If the difference is larger than one frame, the BNF stride or network structure is probably incorrect.
            raise RuntimeError(
                f"BNF length mismatch too large before match: "
                f"bnf_feat={bnf_feat.shape[-1]}, target={T}, bnf_stride={self.bnf_stride}"
            )

        bnf_feat = self._match_time_length(bnf_feat, T) # Correct the remaining at-most-one-frame mismatch so the BNF length exactly equals the x length.

        net_in = torch.cat([x, cond_expanded, bnf_feat], dim=1) # Concatenate x, the condition features, and the BNF feature along the channel dimension.
        out = self.conv(net_in) # Send the combined feature into this block's main convolution layer.
        out = F.glu(out, dim=1) # Apply GLU, which splits the channels into a content half and a gate half.
        return out # Return the output feature produced by this block.


# 4. Define the complete VoiceGrad model.
class VoiceGrad(nn.Module):
    def __init__(
        self,
        n_mels=80, # Number of mel-spectrogram channels.
        n_bnf=144, # Original BNF feature dimension.
        n_channels=512, # Hidden channel width of the U-Net backbone.
        n_spk=18, # Number of target speakers represented by the speaker embedding table.
        n_levels=20, # Number of discrete diffusion noise levels.
        cond_dim=128, # Dimension used by both the noise-level embedding and the speaker embedding.
        bnf_out_dim=32 # Number of BNF channels after projection.
    ):
        super().__init__() # Initialize the `nn.Module` parent class so PyTorch can register and manage all submodules and parameters.

        self.n_channels = n_channels # Save the backbone hidden-channel count as a model attribute.

        # 5. Define the two condition embedding layers.
        self.noise_emb = nn.Embedding(n_levels + 1, cond_dim) # Convert each discrete noise-level index into a learnable `cond_dim`-dimensional vector.
        self.spk_emb = nn.Embedding(n_spk, cond_dim) # Convert each discrete speaker index into a learnable `cond_dim`-dimensional vector.

        # 6. Define the encoder part of the U-Net.
        self.layer1 = VoiceGradBlock(
            n_mels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        ) # Input length T and output length T.

        self.layer2 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        ) # Current input length T; the main convolution downsamples the output to approximately T/2.

        self.layer3 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        ) # Current input length T/2 and output length T/2.

        self.layer4 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        ) # Current input length T/2; the main convolution downsamples the output to approximately T/4.

        self.layer5 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        ) # Current input length T/4 and output length T/4.

        self.layer6 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        ) # Current input length T/4 and output length T/4.

        # 7. Define the decoder part of the U-Net.
        self.layer7 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        ) # Current input length T/4 and output length T/4.

        self.layer8 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4, transpose=True
        ) # Transposed-convolution upsampling: current input length T/4 and output length approximately T/2.

        self.layer9 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        ) # Current input length T/2 and output length T/2.

        self.layer10 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2, transpose=True
        ) # Transposed-convolution upsampling: current input length T/2 and output length approximately T.

        self.layer11 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        ) # Current input length T and output length T.

        # 8. Define the final output layer that converts 512 hidden channels back to the 80-channel mel-noise prediction.
        self.final_conv = nn.utils.weight_norm(
            nn.Conv1d(
                n_channels, # Input channel count: 512 hidden channels.
                n_mels, # Output channel count: 80 mel channels.
                kernel_size=9, # Use a kernel size of 9 to process local temporal context.
                padding=4 # Add symmetric padding so the time length remains unchanged.
            )
        )

        self.apply(self._init_weights) # Apply the weight-initialization function defined below to every registered module in the model.

    # 9. Define the weight-initialization rule for convolution layers.
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)): # Check whether the current module is a normal 1D convolution or a transposed 1D convolution.
            nn.init.xavier_normal_(m.weight, gain=0.5) # Initialize the convolution weight with Xavier normal initialization using gain=0.5.
            if m.bias is not None: # If this layer has a bias parameter.
                nn.init.constant_(m.bias, 0) # Initialize every bias value to zero.

    # 10. Match the time lengths of the two tensors used in a skip connection.
    def _match_shape(self, x, target):
        if x.shape[-1] < target.shape[-1]: # If x is shorter than the target tensor.
            diff = target.shape[-1] - x.shape[-1] # Calculate the difference in time frames.
            x = F.pad(x, (0, diff)) # Pad zeros on the right side of x.
        elif x.shape[-1] > target.shape[-1]: # If x is longer than the target tensor.
            x = x[..., :target.shape[-1]] # Crop the extra frames from the right side.
        return x # Return x after its time length has been matched to the target tensor.

    # 11. Define the complete forward pass of VoiceGrad.
    def forward(self, x, noise_level, speaker_id, bnf=None):
        if bnf is None: # If no BNF conditioning feature was provided.
            raise ValueError("BNF-conditioned VoiceGrad requires bnf input, but got None.") # Raise an error immediately because this is the BNF-conditioned version of VoiceGrad.

        # 12. Convert the noise-level indices and speaker indices into learnable vectors.
        n_emb = self.noise_emb(noise_level).unsqueeze(-1) # Shape change: [B] -> [B, cond_dim] -> [B, cond_dim, 1].
        s_emb = self.spk_emb(speaker_id).unsqueeze(-1) # Shape change: [B] -> [B, cond_dim] -> [B, cond_dim, 1].
        cond = torch.cat([n_emb, s_emb], dim=1) # Concatenate along the channel dimension -> [B, 2*cond_dim, 1].

        # 13. Encoder: progressively extract features and perform two temporal downsampling steps.
        out1 = self.layer1(x, cond, bnf) # Time length: T -> T.
        out2 = self.layer2(out1, cond, bnf) # Time length: T -> T/2.
        out3 = self.layer3(out2, cond, bnf) # Time length: T/2 -> T/2.
        out4 = self.layer4(out3, cond, bnf) # Time length: T/2 -> T/4.
        out5 = self.layer5(out4, cond, bnf) # Time length: T/4 -> T/4.
        out6 = self.layer6(out5, cond, bnf) # Time length: T/4 -> T/4.

        # 14. Decoder: progressively restore the time length and add skip connections from the encoder.
        out7 = self.layer7(out6, cond, bnf) # Time length: T/4 -> T/4.
        out7 = self._match_shape(out7, out5) + out5 # Match the shape to encoder feature out5 and add them to form a skip connection.

        out8 = self.layer8(out7, cond, bnf) # Time length: T/4 -> T/2.
        out8 = self._match_shape(out8, out3) + out3 # Match the shape to encoder feature out3 and add them.

        out9 = self.layer9(out8, cond, bnf) # Time length: T/2 -> T/2.
        out9 = self._match_shape(out9, out2) + out2 # Match the shape to encoder feature out2 and add them.

        out10 = self.layer10(out9, cond, bnf) # Time length: T/2 -> T.
        out10 = self._match_shape(out10, out1) + out1 # Match the shape to encoder feature out1 and add them.

        out11 = self.layer11(out10, cond, bnf) # Time length: T -> T.

        # 15. Convert the final 512-channel hidden feature into an 80-channel noise prediction.
        out = self.final_conv(out11) # Shape change: [B, 512, T] -> [B, 80, T].
        return out # Return epsilon_theta, which is the noise predicted by the model.


# 16. Run the following self-test only when this model file is executed directly.
if __name__ == '__main__':
    batch_size = 2 # Use a batch size of 2 for the self-test.
    n_mels = 80 # Number of mel channels.
    time_steps = 161 # Intentionally use an odd time length to verify that the model handles odd numbers of frames correctly.
    n_bnf = 144 # BNF feature dimension.

    x = torch.randn(batch_size, n_mels, time_steps) # Generate a random fake mel input with shape [2, 80, 161].
    bnf = torch.randn(batch_size, n_bnf, time_steps) # Generate a random fake BNF input with shape [2, 144, 161].

    noise_idx = torch.randint(0, 20, (batch_size,)) # Randomly generate one noise-level index for each sample.
    spk_idx = torch.randint(0, 4, (batch_size,)) # Randomly generate one target-speaker index for each sample.

    model = VoiceGrad(n_mels=80, n_bnf=144, n_channels=512, n_spk=4) # Create the complete VoiceGrad model.
    output = model(x, noise_idx, spk_idx, bnf=bnf) # Run one forward pass through the model.

    print(f"Input: {x.shape}, BNF: {bnf.shape}, Output: {output.shape}") # Print the input, BNF, and output shapes for inspection.
    assert x.shape == output.shape # Assert that the model output shape is exactly the same as the input mel shape.
    print("Verification Passed: model handles odd lengths and aligned BNF correctly.") # Print a success message after the shape verification passes.
