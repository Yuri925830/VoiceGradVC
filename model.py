import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceGrad(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_bnf=144,
        n_channels=512,
        n_spk=18,
        n_levels=20,
        cond_dim=128,
        bnf_out_dim=32
    ):
        """
        VoiceGrad Score Approximator Model.

        Key fixes made:
        1. In dataset.py, input BNF features have been aligned to the original time length T of mel-spectrograms.
        2. For each layer in this model, we select an appropriate BNF stride based on the input length of that layer.
        3. No longer use F.interpolate to force-align BNF features (this was a temporary workaround before).
        """
        super().__init__()

        self.n_channels = n_channels

        # Embedding layers for conditional inputs
        self.noise_emb = nn.Embedding(n_levels + 1, cond_dim)  # Embedding for noise levels
        self.spk_emb = nn.Embedding(n_spk, cond_dim)           # Embedding for speaker IDs

        # =========================
        # Design principle for BNF stride (determined by input length of each layer)
        #
        # layer1 input length: T    -> bnf_stride = 1
        # layer2 input length: T    -> bnf_stride = 1
        # layer3 input length: T/2  -> bnf_stride = 2
        # layer4 input length: T/2  -> bnf_stride = 2
        # layer5 input length: T/4  -> bnf_stride = 4
        # layer6 input length: T/4  -> bnf_stride = 4
        # layer7 input length: T/4  -> bnf_stride = 4
        # layer8 input length: T/4  -> bnf_stride = 4
        # layer9 input length: T/2  -> bnf_stride = 2
        # layer10 input length: T/2 -> bnf_stride = 2
        # layer11 input length: T   -> bnf_stride = 1
        #
        # This aligns with the paper's requirement: "stride r should be compatible with the input length of the layer"
        # =========================

        # Encoder layers
        self.layer1 = VoiceGradBlock(
            n_mels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )
        self.layer2 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )  # input length = T
        self.layer3 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input length = T/2
        self.layer4 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input length = T/2
        self.layer5 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input length = T/4
        self.layer6 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input length = T/4

        # Decoder layers
        self.layer7 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input length = T/4
        self.layer8 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4, transpose=True
        )  # input length = T/4 (transpose for upsampling)
        self.layer9 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input length = T/2
        self.layer10 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2, transpose=True
        )  # input length = T/2 (transpose for upsampling)
        self.layer11 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )  # input length = T

        # Final convolution layer to map back to mel-spectrogram dimensions
        self.final_conv = nn.utils.weight_norm(
            nn.Conv1d(n_channels, n_mels, kernel_size=9, padding=4)
        )

        # Initialize weights for all layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Custom weight initialization function for convolutional layers.
        Used xavier normal initialization for weights (gain=0.5) and zero for biases.
        """
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _match_shape(self, x, target):
        """
        Align the length of tensor x to match the target tensor (for skip connections).
        Only do minimal padding/cropping, no interpolation is used here.
        """
        if x.shape[-1] < target.shape[-1]:
            diff = target.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff))  # Pad the last dimension (time axis)
        elif x.shape[-1] > target.shape[-1]:
            x = x[..., :target.shape[-1]]  # Crop to target length
        return x

    def forward(self, x, noise_level, speaker_id, bnf=None):
        """
        Forward pass of the VoiceGrad model.
        Args:
            x: Input mel-spectrogram (batch_size, n_mels, time_steps)
            noise_level: Noise level indices for embedding (batch_size,)
            speaker_id: Speaker IDs for embedding (batch_size,)
            bnf: BNF linguistic features (batch_size, n_bnf, time_steps)
        Returns:
            Output mel-spectrogram (batch_size, n_mels, time_steps)
        """
        if bnf is None:
            raise ValueError("BNF-conditioned VoiceGrad requires bnf input, but got None.")

        # Get noise and speaker embeddings, expand to time dimension (add length=1 for broadcasting)
        n_emb = self.noise_emb(noise_level).unsqueeze(-1)   # Shape: [B, cond_dim, 1]
        s_emb = self.spk_emb(speaker_id).unsqueeze(-1)      # Shape: [B, cond_dim, 1]
        cond = torch.cat([n_emb, s_emb], dim=1)             # Combine embeddings: [B, 2*cond_dim, 1]

        # -------- Encoder Forward --------
        out1 = self.layer1(x, cond, bnf)    # [B, C, T]
        out2 = self.layer2(out1, cond, bnf) # [B, C, T/2] (downsampled by layer2 stride=2)
        out3 = self.layer3(out2, cond, bnf) # [B, C, T/2]
        out4 = self.layer4(out3, cond, bnf) # [B, C, T/4] (downsampled by layer4 stride=2)
        out5 = self.layer5(out4, cond, bnf) # [B, C, T/4]
        out6 = self.layer6(out5, cond, bnf) # [B, C, T/4]

        # -------- Decoder Forward (with skip connections) --------
        out7 = self.layer7(out6, cond, bnf)
        out7 = self._match_shape(out7, out5) + out5  # Skip connection from out5

        out8 = self.layer8(out7, cond, bnf)
        out8 = self._match_shape(out8, out3) + out3  # Skip connection from out3

        out9 = self.layer9(out8, cond, bnf)
        out9 = self._match_shape(out9, out2) + out2  # Skip connection from out2

        out10 = self.layer10(out9, cond, bnf)
        out10 = self._match_shape(out10, out1) + out1  # Skip connection from out1

        out11 = self.layer11(out10, cond, bnf)

        # Final convolution to get output mel-spectrogram
        out = self.final_conv(out11)
        return out


class VoiceGradBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        s,
        cond_dim,
        bnf_dim,
        bnf_out,
        bnf_stride=1,
        transpose=False
    ):
        """
        Basic building block for VoiceGrad model (encoder/decoder layer).
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            k: Kernel size of convolution
            s: Stride of convolution
            cond_dim: Dimension of conditional embeddings (noise + speaker)
            bnf_dim: Dimension of input BNF features
            bnf_out: Dimension of projected BNF features
            bnf_stride: Stride for BNF projection convolution
            transpose: If True, use ConvTranspose1d (for upsampling/decoder); else Conv1d (encoder)
        """
        super().__init__()
        self.transpose = transpose
        self.bnf_stride = bnf_stride

        # Key fix:
        # Original code used the same kernel as main conv plus incorrect stride_factor,
        # then relied on F.interpolate to fix length mismatch (a workaround).
        # 
        # Now we use a dedicated strided convolution for BNF, only to reduce time resolution 
        # to match the input length of current layer.
        # Benefits of kernel_size=1:
        # - Avoids losing one frame when using even kernel size with stride=1
        # - Compatible with paper's requirement of "32 channels + stride r"
        self.bnf_proj = nn.utils.weight_norm(
            nn.Conv1d(
                bnf_dim,
                bnf_out,
                kernel_size=1,
                stride=bnf_stride,
                padding=0
            )
        )

        # Total input channels for main convolution: 
        # input channels + conditional embedding channels + projected BNF channels
        total_in_ch = in_ch + (cond_dim * 2) + bnf_out
        glu_out_ch = out_ch * 2  # GLU splits channels by 2, so output is out_ch

        if transpose:
            # Deconvolution / Upsampling (for decoder layers)
            padding = (k - s) // 2
            self.conv = nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    total_in_ch,
                    glu_out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding
                )
            )
        else:
            # Convolution / Keep/Downsampling (for encoder layers)
            padding = (k - 1) // 2  # Same padding to keep length (when stride=1)
            self.conv = nn.utils.weight_norm(
                nn.Conv1d(
                    total_in_ch,
                    glu_out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding
                )
            )

    def _match_time_length(self, x, target_len):
        """
        Align the time dimension of tensor x to target_len.
        Only do minimal padding/cropping, no interpolation (unlike the old workaround).
        """
        if x.shape[-1] < target_len:
            diff = target_len - x.shape[-1]
            x = F.pad(x, (0, diff))  # Pad time axis (right side)
        elif x.shape[-1] > target_len:
            x = x[..., :target_len]  # Crop time axis to target length
        return x

    def forward(self, x, cond, bnf):
        """
        Forward pass of VoiceGradBlock.
        Args:
            x: Input feature tensor (batch_size, in_ch, time_steps)
            cond: Combined noise+speaker embedding (batch_size, 2*cond_dim, 1)
            bnf: BNF features (batch_size, bnf_dim, time_steps)
        Returns:
            Output tensor after convolution and GLU (batch_size, out_ch, time_steps)
        """
        if bnf is None:
            raise ValueError("VoiceGradBlock requires bnf input, but got None.")

        # Get time length of current input (for alignment)
        T = x.shape[-1]

        # Expand conditional embedding to match the time length of input x
        cond_expanded = cond.expand(-1, -1, T)

        # Project BNF features and adjust time resolution with pre-defined stride
        bnf_feat = self.bnf_proj(bnf)

        # Normally, the length should be almost compatible (only 1 frame difference for odd lengths)
        # Raise error if mismatch is too large (indicates wrong stride setting)
        if abs(bnf_feat.shape[-1] - T) > 1:
            raise RuntimeError(
                f"BNF length mismatch too large before match: "
                f"bnf_feat={bnf_feat.shape[-1]}, target={T}, bnf_stride={self.bnf_stride}"
            )

        # Align BNF feature length to input x's time length
        bnf_feat = self._match_time_length(bnf_feat, T)

        # Concatenate input, conditional embedding, and projected BNF along channel dimension
        net_in = torch.cat([x, cond_expanded, bnf_feat], dim=1)
        out = self.conv(net_in)
        out = F.glu(out, dim=1)  # Gated Linear Unit (split channels by 2 along dim=1)
        return out


if __name__ == '__main__':
    # Simple self-test: test model with odd time length input (to check alignment)
    batch_size = 2
    n_mels = 80
    time_steps = 161  # Odd length (easy to catch padding/cropping bugs)
    n_bnf = 144

    # Random input mel-spectrogram (batch_size, n_mels, time_steps)
    x = torch.randn(batch_size, n_mels, time_steps)
    # Key point: After dataset fix, input BNF has the same time length as original mel
    bnf = torch.randn(batch_size, n_bnf, time_steps)

    # Random noise level and speaker ID (batch_size,)
    noise_idx = torch.randint(0, 20, (batch_size,))
    spk_idx = torch.randint(0, 4, (batch_size,))

    # Initialize model and forward pass
    model = VoiceGrad(n_mels=80, n_bnf=144, n_channels=512, n_spk=4)
    output = model(x, noise_idx, spk_idx, bnf=bnf)

    # Print shapes and verify input/output match
    print(f"Input: {x.shape}, BNF: {bnf.shape}, Output: {output.shape}")
    assert x.shape == output.shape
    print("Verification Passed: model handles odd lengths and aligned BNF correctly.")