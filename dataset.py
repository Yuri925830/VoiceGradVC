import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import re


class VoiceGradDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split='train',
        segment_length=128,
        sample_rate=16000,
        hop_size=256,
        bnf_frame_shift_ms=10.0
    ):
        """
        Dataset class for VoiceGrad model (mel-spectrogram + BNF linguistic features).
        Args:
            root_dir: Root directory of the dataset (contains mel/ and bnf/ subfolders)
            split: Dataset split ('train'/'val'/'test')
            segment_length: Length of audio segments (only for training, None for val/test)
            sample_rate: Audio sample rate (16000 Hz for CMU Arctic)
            hop_size: Hop size for mel-spectrogram extraction (256 samples)
            bnf_frame_shift_ms: Frame shift for BNF features (10 ms per frame)
        """
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.mel_dir = os.path.join(root_dir, 'mel')  # Directory for mel-spectrogram files
        self.bnf_dir = os.path.join(root_dir, 'bnf')  # Directory for BNF feature files
        self.split = split

        # ===== Time axis parameters =====
        # mel-spectrogram: 16000 Hz / hop_size=256 -> 62.5 frames per second (fps)
        # BNF: frame_shift=10ms -> 100 frames per second (fps)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_fps = sample_rate / hop_size
        self.bnf_frame_shift_ms = bnf_frame_shift_ms
        self.bnf_fps = 1000.0 / bnf_frame_shift_ms  # Convert ms to fps

        # Closed-set speakers (for training/target) - K=4
        self.train_speakers = ['clb', 'bdl', 'slt', 'rms']
        # Open-set speakers (only for test source)
        self.openset_speakers = ['jmk', 'ksp', 'lnh']

        # Fixed speaker ID mapping (train speakers only)
        self.spk2id = {spk: i for i, spk in enumerate(self.train_speakers)}

        # Load mel-spectrogram statistics (mean/std for normalization)
        stats_dir = os.path.join(root_dir, 'stats')
        if not os.path.exists(os.path.join(stats_dir, 'mel_mean.npy')):
            print("【Warning】No statistics files found, using temporary mean/std...")
            self.mel_mean = torch.zeros(80, 1)
            self.mel_std = torch.ones(80, 1)
        else:
            # Load precomputed mean/std and reshape to [80, 1] (per mel bin)
            self.mel_mean = torch.from_numpy(
                np.load(os.path.join(stats_dir, 'mel_mean.npy'))
            ).float().view(-1, 1)
            self.mel_std = torch.from_numpy(
                np.load(os.path.join(stats_dir, 'mel_std.npy'))
            ).float().view(-1, 1)

        # List to store valid file paths (mel + bnf) and speaker info
        self.file_list = []

        # Get all speaker directories in mel folder
        all_speakers = sorted([
            d for d in os.listdir(self.mel_dir)
            if os.path.isdir(os.path.join(self.mel_dir, d))
        ])

        debug_counter = 0  # For validation set debugging (print first 3 samples)

        for spk in all_speakers:
            spk_mel_dir = os.path.join(self.mel_dir, spk)
            spk_bnf_dir = os.path.join(self.bnf_dir, spk)

            # Get all .npy mel files for current speaker
            files = [f for f in os.listdir(spk_mel_dir) if f.endswith('.npy')]
            files.sort()

            for f in files:
                try:
                    # Extract numerical index from filename (for split selection)
                    nums = re.findall(r'\d+', f)
                    local_idx = int(nums[-1])

                    # CMU Arctic dataset: Set A = 593 utterances, Set B = 539 utterances
                    # Global index = local index + 593 for Set B files
                    if 'arctic_b' in f or '_b' in f:
                        global_idx = local_idx + 593
                    else:
                        global_idx = local_idx

                except Exception:
                    print(f"Warning: Skip {f}, parse error.")
                    continue

                # Check if current file belongs to the selected split (train/val/test)
                is_valid = self._is_file_in_split(spk, global_idx, split)

                # Print debug info for first 3 validation samples
                if split == 'val' and is_valid and debug_counter < 3:
                    print(f"[Val Debug] Keep {spk} {f} -> Global Idx {global_idx}")
                    debug_counter += 1

                if is_valid:
                    # Generate BNF filename (replace .npy with .ling_feat.npy)
                    bnf_name = f.replace('.npy', '.ling_feat.npy')

                    # Fallback: if BNF file with ling_feat suffix not found, use same name as mel
                    if not os.path.exists(os.path.join(spk_bnf_dir, bnf_name)):
                        bnf_name = f

                    bnf_path = os.path.join(spk_bnf_dir, bnf_name)
                    # Only add to file list if BNF file exists
                    if os.path.exists(bnf_path):
                        # Assign speaker ID (-1 for open-set speakers not in train_speakers)
                        spk_id = self.spk2id[spk] if spk in self.spk2id else -1
                        self.file_list.append({
                            'mel_path': os.path.join(spk_mel_dir, f),
                            'bnf_path': bnf_path,
                            'spk_id': spk_id,
                            'spk_name': spk,
                            'global_idx': global_idx
                        })

        # Print dataset statistics
        print(f"Dataset split: {split} | Samples: {len(self.file_list)}")
        print(
            f"[Time Axis] mel_fps={self.mel_fps:.4f}, "
            f"bnf_fps={self.bnf_fps:.4f}, "
            f"expected_ratio={self.bnf_fps / self.mel_fps:.4f}"
        )

        # Check training set size (expected 1000 samples for CMU Arctic)
        if split == 'train' and len(self.file_list) != 1000:
            print(f"【Note】Training set size is {len(self.file_list)}, expected 1000. Check file completeness.")

    def _is_file_in_split(self, spk, idx, split):
        """
        Determine if a file (by speaker and global index) belongs to the selected split.
        CMU Arctic global index range: 1-1132
        Args:
            spk: Speaker name (e.g., 'clb', 'bdl')
            idx: Global index of the utterance (1-1132)
            split: Target split ('train'/'val'/'test')
        Returns:
            bool: True if file is in the split, False otherwise
        """
        # 1. TEST SET (global index 1101 - 1132)
        # Includes both closed-set (train) and open-set speakers
        if split == 'test':
            if 1101 <= idx <= 1132:
                if spk in self.train_speakers or spk in self.openset_speakers:
                    return True
            return False

        # 2. VALIDATION SET (global index 1001 - 1100)
        # Only includes closed-set train speakers
        if split == 'val':
            if 1001 <= idx <= 1100:
                if spk in self.train_speakers:
                    return True
            return False

        # 3. TRAINING SET (global index 1 - 1000) - non-parallel strict split
        # Each speaker has exactly 250 utterances:
        # clb: 1-250, bdl:251-500, slt:501-750, rms:751-1000
        if split == 'train':
            if not (1 <= idx <= 1000):
                return False

            if spk == 'clb' and (1 <= idx <= 250):
                return True
            if spk == 'bdl' and (251 <= idx <= 500):
                return True
            if spk == 'slt' and (501 <= idx <= 750):
                return True
            if spk == 'rms' and (751 <= idx <= 1000):
                return True

            return False

        return False

    def __len__(self):
        """Return total number of valid samples in the dataset split."""
        return len(self.file_list)

    def _ensure_mel_shape(self, mel):
        """
        Ensure mel-spectrogram has the correct shape [80, T] (80 mel bins, T time steps).
        If shape is [T, 80], transpose it. Raise error for invalid shapes.
        """
        if mel.ndim != 2:
            raise ValueError(f"mel ndim should be 2, but got {mel.shape}")
        if mel.shape[0] == 80:
            return mel
        if mel.shape[1] == 80:
            return mel.T
        raise ValueError(f"Invalid mel shape: {mel.shape}")

    def _ensure_bnf_shape(self, bnf):
        """
        Ensure BNF features have the correct shape [144, T] (144 BNF dims, T time steps).
        If shape is [T, 144], transpose it. Raise error for invalid shapes.
        """
        if bnf.ndim != 2:
            raise ValueError(f"bnf ndim should be 2, but got {bnf.shape}")
        if bnf.shape[0] == 144:
            return bnf
        if bnf.shape[1] == 144:
            return bnf.T
        raise ValueError(f"Invalid bnf shape: {bnf.shape}")

    def _resample_bnf_to_mel_length(self, bnf, target_len):
        """
        Resample BNF features from [144, T_bnf] to [144, target_len] along time axis.
        Solves the 1.6x time axis mismatch between BNF (100fps) and mel (62.5fps).
        Unlike hard cropping (old method), this uses linear interpolation to preserve temporal info.
        Args:
            bnf: Original BNF array (shape [144, T_bnf])
            target_len: Target time length (same as mel-spectrogram)
        Returns:
            Resampled BNF array (shape [144, target_len])
        """
        current_len = bnf.shape[1]

        # No resampling needed if lengths match
        if current_len == target_len:
            return bnf

        # Extreme fallback: repeat BNF if length is too short (avoids interpolation errors)
        if current_len <= 1:
            return np.repeat(bnf, target_len, axis=1)

        # Convert numpy array to torch tensor (add batch dimension for interpolate)
        bnf_tensor = torch.from_numpy(bnf).float().unsqueeze(0)  # Shape: [1, 144, T_bnf]

        # 1D linear interpolation along time axis
        bnf_tensor = F.interpolate(
            bnf_tensor,
            size=target_len,
            mode='linear',
            align_corners=False
        )

        # Convert back to numpy array (remove batch dimension)
        return bnf_tensor.squeeze(0).cpu().numpy()

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        Args:
            idx: Index of the sample in file_list
        Returns:
            Dictionary containing:
                mel: Normalized mel-spectrogram (tensor, [80, segment_length])
                bnf: Resampled BNF features (tensor, [144, segment_length])
                spk_id: Speaker ID (tensor, scalar long)
                spk_name: Speaker name (string)
        """
        item = self.file_list[idx]

        # Load mel and BNF files (retry with random index if load fails)
        try:
            mel = np.load(item['mel_path'])
            bnf = np.load(item['bnf_path'])
        except Exception:
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        # Ensure correct shape for mel and BNF (retry if shape error)
        try:
            mel = self._ensure_mel_shape(mel)
            bnf = self._ensure_bnf_shape(bnf)
        except Exception as e:
            print(f"[Shape Error] {item['mel_path']} / {item['bnf_path']} -> {e}")
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        # ===== Key Fix =====
        # Old method used hard cropping to min(mel_len, bnf_len), hiding the 1.6x mismatch.
        # New method: resample BNF to match mel length first.
        mel_len = mel.shape[1]
        bnf = self._resample_bnf_to_mel_length(bnf, mel_len)

        # Verify mel and BNF have the same time length after resampling
        assert mel.shape[1] == bnf.shape[1], \
            f"Length mismatch after resample: mel={mel.shape}, bnf={bnf.shape}"

        total_len = mel.shape[1]

        # Segment or pad to fixed length (only for training)
        if self.segment_length is not None:
            if total_len > self.segment_length:
                # Randomly crop a segment of segment_length
                start = random.randint(0, total_len - self.segment_length)
                end = start + self.segment_length
                mel = mel[:, start:end]
                bnf = bnf[:, start:end]
            else:
                # Pad with zeros to reach segment_length (right side padding)
                pad_len = self.segment_length - total_len
                mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant')
                bnf = np.pad(bnf, ((0, 0), (0, pad_len)), mode='constant')

        # Convert to torch tensors
        mel_tensor = torch.from_numpy(mel).float()
        bnf_tensor = torch.from_numpy(bnf).float()

        # Normalize mel-spectrogram with precomputed mean/std
        mel_normalized = (mel_tensor - self.mel_mean) / self.mel_std

        return {
            'mel': mel_normalized,
            'bnf': bnf_tensor,
            'spk_id': torch.tensor(item['spk_id']).long(),
            'spk_name': item['spk_name']
        }


def get_dataloader(root_dir, split, batch_size, num_workers=4):
    """
    Create a DataLoader for VoiceGradDataset.
    Args:
        root_dir: Root directory of the dataset
        split: Dataset split ('train'/'val'/'test')
        batch_size: Batch size (train: batch_size, val/test: 1)
        num_workers: Number of worker processes for data loading
    Returns:
        DataLoader object
    """
    # Set segment length (128 for training, None for val/test - full length)
    seg_len = 128 if split == 'train' else None
    dataset = VoiceGradDataset(
        root_dir=root_dir,
        split=split,
        segment_length=seg_len,
        sample_rate=16000,
        hop_size=256,
        bnf_frame_shift_ms=10.0
    )
    # Set batch size (1 for val/test to avoid padding issues with variable lengths)
    bs = batch_size if split == 'train' else 1
    # Shuffle only for training set
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)