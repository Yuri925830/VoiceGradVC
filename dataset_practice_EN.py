# Step 1: import packages
import os # Handles paths and folder traversal
import re # Extracts numeric sentence IDs from filenames
import random # Handles random cropping: during training, utterances have different lengths, so one batch needs a unified length
import numpy as np # Reads .npy files (mel, BNF, mean, std)
import torch # Converts NumPy arrays into Torch tensors
import torch.nn.functional as F # Uses F.interpolate for BNF time-axis resampling
from torch.utils.data import Dataset, DataLoader # Dataset defines how samples are read; DataLoader automatically forms PyTorch batches

# Step 2: create a data reader
class VoiceGradDataset(Dataset):
    def __init__(
        self, # The dataset object itself
        root_dir, # Data root directory
        split='train', # split indicates which part of the data is being read
        segment_length=128, # Randomly crop 128 frames
        sample_rate=16000, # Sampling rate 16000 Hz: 1 second of audio contains 16000 sample points
        hop_size=256, # Hop size, measured in sample points. hop_size / sample_rate = hop_time in ms, meaning a new mel frame is generated every xx ms
        bnf_frame_shift_ms=10.0 # BNF defaults to one frame every 10 milliseconds
    ):
        self.root_dir = root_dir # Save the data root directory
        self.split = split # Save whether the current split is train, val, or test
        self.segment_length = segment_length # Save the training crop length
        self.mel_dir = os.path.join(root_dir,'mel') # Build the mel folder path. x0
        self.bnf_dir = os.path.join(root_dir,'bnf') # Build the BNF folder path. p
        # Time-axis parameters
        # mel: 16k / hop_size=256 -> 62.5 fps
        # bnf: frame_shift=10ms -> 100 fps
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_fps = sample_rate / hop_size # Number of mel frames per second
        self.bnf_frame_shift_ms = bnf_frame_shift_ms
        self.bnf_fps = 1000.0 / bnf_frame_shift_ms # Number of BNF frames per second
        # Closed-set target speakers: used for training and also used as target voice identities
        self.train_speakers = ['clb','bdl','slt','rms']
        # Open-set source speakers for testing
        self.openset_speakers = ['jmk','ksp','lnh']
        # Convert speaker names into numeric IDs, k
        self.spk2id = {spk:i for i,spk in enumerate(self.train_speakers)}
        # Load the paths of the mean and standard-deviation files for normalization
        stats_dir = os.path.join(root_dir,'stats')
        mean_path = os.path.join(stats_dir,'mel_mean.npy')
        std_path = os.path.join(stats_dir,'mel_std.npy')
        # Check whether mel_mean.npy exists
        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"mel_mean.npy not found: {mean_path}")
        if not os.path.exists(std_path):
            raise FileNotFoundError(f"mel_std.npy not found: {std_path}")

        # Read mel_mean.npy and mel_std.npy, convert NumPy arrays into PyTorch tensors, ensure float32 for training, and reshape to [80, 1]
        self.mel_mean = torch.from_numpy(
        	np.load(mean_path)
        ).float().view(-1,1)
        self.mel_std = torch.from_numpy(
            np.load(std_path)
        ).float().view(-1,1)

        # Save all valid sample paths
        self.file_list = []
        # Scan speaker folders
        # First create an all_speakers list, then use sorted() to keep the speaker order fixed
        all_speaks = sorted([
            # Use os.listdir() to list everything under data/mel and iterate over it. Normally, d will be a speaker-name folder
            d for d in os.listdir(self.mel_dir)
            # For safety, keep only folders, and join the mel root directory with the folder name, such as /nfs/speechst2/storage/rich_925/VoiceGrad/data/mel/clb
            if os.path.isdir(os.path.join(self.mel_dir,d))
        ])
        # Iterate over each speaker folder
        for spk in all_speaks:
            # Enter each speaker's mel / BNF folder
            spk_mel_dir = os.path.join(self.mel_dir,spk)
            spk_bnf_dir = os.path.join(self.bnf_dir,spk)
            # List all .npy files in the current speaker's mel folder
            files = [f for f in os.listdir(spk_mel_dir) if f.endswith('.npy')] # Here we are already inside a specific speaker's mel folder, so we use spk_mel_dir, not the root directory self.mel_dir
            # Sort files
            files.sort()
            # Process each mel file one by one
            for f in files:
                try:
                    # Find all consecutive digits in the filename, for example: arctic_a0001.npy -> ["0001"]
                    nums = re.findall(r'\d+',f) # r'' treats escape characters as ordinary characters; \d means any digit, + means the previous element may appear one or more times
                    # Take the last group of digits and convert it to an integer, for example: "0001" -> 1
                    local_idx = int(nums[-1])
                    # For the global ID, arctic_a uses the original ID, while arctic_b adds 593
                    if 'srctic_b' in f or '_b' in f:
                        global_idx = local_idx + 593
                    else:
                        global_idx = local_idx

                except Exception:
                    # If filename parsing fails, skip it and continue with the next file
                    print(f'warning:skip{f},parse error.')
                    continue

                is_valid = self._is_file_in_split(spk,global_idx,split) # Call _is_file_in_split() to judge whether the current file belongs to the current dataset split

                if split == 'val' and is_valid and debug_counter < 3: # If this is the validation set, this file is valid, and fewer than 3 debug messages have been printed, print it for checking
                    print(f'keep {spk} {f} -> Global idx {global_idx}') # Print the kept validation sample to confirm that ID parsing is correct
                    debug_counter += 1 # Increase the print counter by 1
                
                if is_valid:
                    bnf_name = f.replace('.npy', '.ling_feat.npy') # Infer the BNF filename from the mel filename

                    if not os.path.exists(os.path.join(spk_bnf_dir, bnf_name)): # Fallback logic to support two possible save formats
                        bnf_name = f

                    bnf_path = os.path.join(spk_bnf_dir, bnf_name) # Build the full BNF path, such as data/bnf/clb/arctic_a0001.ling_feat.npy
                    if os.path.exists(bnf_path): # Only add this sample to the dataset if the BNF file really exists
                        spk_id = self.spk2id[spk] if spk in self.spk2id else -1 # Convert the speaker name into a numeric ID; open-set speakers are assigned -1
                        self.file_list.append({ # Add this valid sample to the data list
                            'mel_path': os.path.join(spk_mel_dir, f),
                            'bnf_path': bnf_path,
                            'spk_id': spk_id,
                            'spk_name': spk,
                            'global_idx': global_idx # Save the global sentence ID to confirm whether the train / val / test split is correct
                        })
        print(f"Dataset split: {split} | Samples: {len(self.file_list)}") # Print the current split and sample count, for example Dataset split: train | Samples: 1000
        print(
            f"[Time Axis] mel_fps={self.mel_fps:.4f}, " # Print the mel frame rate. 16000 / 256 = 62.5 fps
            f"bnf_fps={self.bnf_fps:.4f}, " # Print the BNF frame rate. 1000 / 10 = 100 fps
            f"expected_ratio={self.bnf_fps / self.mel_fps:.4f}" # Print the BNF-to-mel frame-rate ratio. 100 / 62.5 = 1.6
        )

        if split == 'train' and len(self.file_list) != 1000:
            print(f"[Notice] The training set contains {len(self.file_list)} samples, but 1000 are expected. Please check file completeness.") # If this is the training set and the sample count is not 1000, print a warning

    def _is_file_in_split(self, spk, idx, split): # Judge whether the current file belongs to the current dataset split

        if split == 'test': # If we are building the test set, enter the test-set rule
            if 1101 <= idx <= 1132: # Only sentences 1101 to 1132 can enter the test set
                if spk in self.train_speakers or spk in self.openset_speakers: # The test set may contain 7 speakers: closed-set and open-set speakers
                    return True
            return False
        
        if split == 'val': # If we are building the validation set, enter the validation-set rule
            if 1001 <= idx <= 1100: # The validation set only keeps sentences 1001 to 1100
                if spk in self.train_speakers: # The validation set only uses the 4 closed-set target speakers
                    return True
            return False
        
        if split == 'train': # If we are building the training set, enter the training-set rule
            if not (1 <= idx <= 1000): # The training set only allows sentences 1 to 1000
                return False
            
            if spk == 'clb' and (1 <= idx <= 250): # non-parallel
                return True
            
            if spk == 'bdl' and (251 <= idx <= 500):
                return True
            
            if spk == 'slt' and (501 <= idx <= 750):
                return True
            
            if spk == 'rms' and (751 <= idx <= 1000):
                return True
            
            return False
        return False
    
    def __len__(self): # Calculate how many samples the dataset has
        return len(self.file_list)
    
    def _ensure_mel_shape(self, mel): # Ensure that the loaded mel always becomes shape [80, T]
        if mel.ndim != 2: # Check whether mel is a 2D array
            raise ValueError(f"mel ndim should be 2, but got {mel.shape}") # If mel is not 2D, actively raise an error
        if mel.shape[0] == 80: # If dimension 0 is 80, the shape is already correct, so return it directly
            return mel
        if mel.shape[1] == 80: # If dimension 1 is 80, the current shape is [T, 80], so it must be transposed
            return mel.T
        raise ValueError(f"Invalid mel shape: {mel.shape}") # If neither the first nor second dimension is 80, this file does not look like a mel feature
    
    def _ensure_bnf_shape(self, bnf): # Ensure that the loaded BNF always becomes shape [144, T]
        if bnf.ndim != 2: # Check whether BNF is a 2D array
            raise ValueError(f"bnf ndim should be 2, but got {bnf.shape}") # If it is not 2D, actively raise an error
        if bnf.shape[0] == 144: # If dimension 0 is already 144, it is already [144, T], so return it directly
            return bnf
        if bnf.shape[1] == 144: # If dimension 1 is 144, it is currently [T, 144], so use .T to transpose it to [144, T]
            return bnf.T
        raise ValueError(f"Invalid bnf shape: {bnf.shape}") # If neither dimension is 144, the file does not look like the BNF we need
    
    def _resample_bnf_to_mel_length(self, bnf, target_len): # Change the time length of BNF to match mel
        current_len = bnf.shape[1] # Get the current BNF time length
        if current_len == target_len:
            return bnf # If BNF is already as long as mel, return it directly without processing
        if current_len <= 1:
            return np.repeat(bnf, target_len, axis=1) # If BNF has only one frame or fewer, linear interpolation is meaningless, so repeat this frame along the time axis to target_len
        bnf_tensor = torch.from_numpy(bnf).float().unsqueeze(0) # Convert the NumPy array into a PyTorch tensor: [144, T_bnf] -> [1, 144, T_bnf]. F.interpolate() for 1D sequences expects input shaped [N, C, L], i.e., batch, channel, length

        bnf_tensor = F.interpolate( # Perform linear interpolation along the BNF time axis
            bnf_tensor,
            size=target_len,
            mode='linear',
            align_corners=False
        )

        return bnf_tensor.squeeze(0).cpu().numpy() # Convert [1, 144, target_len] back to [144, target_len] and then back to a NumPy array, because __getitem__() will continue using NumPy for cropping and padding
    
    def __getitem__(self, idx): # Read one sample and organize it into a format the model can train on
        item = self.file_list[idx] # Take the idx-th sample information from the sample list
        try: # Start trying to read files
            mel = np.load(item['mel_path']) # Read the mel .npy file according to mel_path
            bnf = np.load(item['bnf_path']) # Read the BNF .npy file according to bnf_path
        except Exception: # If reading mel or BNF fails, enter here
            return self.__getitem__(random.randint(0, len(self.file_list) - 1)) # Randomly switch to another sample and read again
        

        try:
            mel = self._ensure_mel_shape(mel)
            bnf = self._ensure_bnf_shape(bnf)
        except Exception as e: # If mel or BNF shape checking fails, enter here. as e stores the detailed error message in variable e
            print(f"[Shape Error] {item['mel_path']} / {item['bnf_path']} -> {e}") # Print which mel file and which BNF file failed, plus the reason
            return self.__getitem__(random.randint(0, len(self.file_list) - 1)) # If this sample is bad, randomly switch to another sample
        
        mel_len = mel.shape[1] # Get the mel time length
        bnf = self._resample_bnf_to_mel_length(bnf, mel_len) # Change the BNF time length to match mel
        # Check whether mel and BNF have exactly the same time length. If this condition is false, the program immediately raises an error
        assert mel.shape[1] == bnf.shape[1], \
            f"Length mismatch after resample: mel={mel.shape}, bnf={bnf.shape}"
        
        total_len = mel.shape[1] # Get the mel time length

        if self.segment_length is not None: # Judge whether cropping is needed. For the training set, self.segment_length = 128, so this branch is used. For validation or test, self.segment_length = None, so full length is kept
            if total_len > self.segment_length: # If the current utterance is longer than 128 frames, perform random cropping
                start = random.randint(0, total_len - self.segment_length) # Randomly choose a crop start point
                end = start + self.segment_length # Calculate the crop end point
                mel = mel[:, start:end] # Crop mel. : keeps all 80 mel channels; start:end takes only this time segment
                bnf = bnf[:, start:end] # Crop BNF with exactly the same start and end. Since mel and BNF are already time-aligned, they must be cropped over the same segment
            else: # If the current utterance is shorter than or equal to 128 frames, it cannot be cropped and must be padded
                pad_len = self.segment_length - total_len # Calculate how many frames are missing
                mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant') # Pad mel with zeros only at the end of the time dimension
                bnf = np.pad(bnf, ((0, 0), (0, pad_len)), mode='constant') # Pad BNF with zeros in the same way

        mel_tensor = torch.from_numpy(mel).float() # Convert NumPy mel into a PyTorch tensor. Deep learning models usually train with float32
        bnf_tensor = torch.from_numpy(bnf).float() # Convert NumPy BNF into a PyTorch tensor
        mel_normalized = (mel_tensor - self.mel_mean) / self.mel_std # Normalize each mel channel separately
        # Return a dictionary. PyTorch DataLoader will automatically combine many such dictionaries into a batch
        return { 
            'mel': mel_normalized, # Return the normalized mel. In diffusion-model training, x_0 is this normalized mel
            'bnf': bnf_tensor,
            'spk_id': torch.tensor(item['spk_id']).long(), # Return the numeric target-speaker ID. .long() converts it to torch.long because the speaker embedding in model.py requires integer indices
            'spk_name': item['spk_name']
        }

def get_dataloader(root_dir, split, batch_size, num_workers=4): # Provide a convenient entry point for external code: create the train / val / test DataLoader in one line. num_workers means how many subprocesses are used for data loading
    seg_len = 128 if split == 'train' else None # If this is the training set, crop to 128 frames
    dataset = VoiceGradDataset( # Create the actual dataset object
        root_dir=root_dir,
        split=split,
        segment_length=seg_len,
        sample_rate=16000,
        hop_size=256,
        bnf_frame_shift_ms=10.0
    )
    bs = batch_size if split == 'train' else 1 # The training set uses the given batch_size, such as 16. Validation and test force batch size to 1
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
