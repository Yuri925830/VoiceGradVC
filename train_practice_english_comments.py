# Import os; it handles path joining, directory creation, and other file-system operations.
import os
# Import random; it provides Python's built-in random functions, and we will fix its seed to make experiments easier to reproduce.
import random
# Import NumPy as np; training uses it to calculate average losses and save history data.
import numpy as np
# Import the main PyTorch library; training, gradients, GPU use, and model saving all depend on it.
import torch
# Import the functional interface as F; this file mainly uses F.l1_loss to compute the L1 noise-prediction loss required by the paper.
import torch.nn.functional as F
# Import matplotlib.pyplot as plt; it is used to draw training and validation loss curves.
import matplotlib.pyplot as plt
# Import tqdm from the tqdm package; it adds progress bars to DataLoaders so we can see batch-level training progress.
from tqdm import tqdm

# Import VoiceGrad from model.py; this is the main neural network that predicts noise epsilon_theta.
from model import VoiceGrad
# Import VoiceGradDiffusion from diffusion.py; it handles forward noising and reverse sampling.
from diffusion import VoiceGradDiffusion
# Import get_dataloader from dataset.py; it creates the training and validation DataLoaders.
from dataset import get_dataloader


# Store all training settings in one dictionary; dictionary syntax is {key: value}, and values are later read with CONFIG['key'].
CONFIG = {
    # data_root must point to the data directory that contains the mel, bnf, and stats subdirectories.
    'data_root': '/content/drive/MyDrive/VoiceGrad/data',

    # Set the total number of training epochs to 4000; one epoch means seeing the complete training set once.
    'epochs': 4000,
    # batch_size=16 means that each optimization step uses 16 training segments together to compute one loss and one set of gradients.
    'batch_size': 16,
    # lr means learning rate; 1e-4 is 0.0001 and controls how large each parameter update is.
    'lr': 1e-4,
    # num_workers=4 lets the DataLoader use four worker processes in parallel, helping it feed data faster.
    'num_workers': 4,
    # seed is the random seed; fixing it makes random cropping, noise generation, and other random operations more reproducible.
    'seed': 1234,

    # Clip the total gradient norm when it exceeds 1.0 to prevent sudden gradient explosion.
    'grad_clip_max_norm': 1.0,
    # Print an extra warning when the pre-clipping gradient norm exceeds 5.0 so unstable training can be noticed early.
    'grad_warn_threshold': 5.0,

    # Run validation every 20 epochs instead of every epoch to avoid spending too much time on validation.
    'val_every': 20,
    # Redraw the loss curve every 50 epochs.
    'plot_every': 50,
    # Save a permanent historical checkpoint every 500 epochs so later checkpoint sweeping is possible.
    'save_every': 500,

    # Directory used to save models, plots, and training history.
    'save_dir': './checkpoints',

    # Ternary expression syntax is A if condition else B; use the GPU when CUDA is available, otherwise fall back to the CPU.
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
# This closing brace ends the CONFIG dictionary.
}


# Define set_seed; the type annotation seed: int indicates that an integer is expected, but it does not change how the function runs.
def set_seed(seed: int):
    # Fix the random state of Python's random module; random cropping in dataset.py is affected by it.
    random.seed(seed)
    # Fix NumPy's random state so later NumPy random operations can also behave more consistently.
    np.random.seed(seed)
    # Fix PyTorch's CPU random state; torch.randn and torch.randint are affected by it.
    torch.manual_seed(seed)
    # Check whether CUDA is actually available; CUDA random seeds only need to be set when a GPU is present.
    if torch.cuda.is_available():
        # Set the same random seed for every visible GPU so multi-GPU environments are also as consistent as possible.
        torch.cuda.manual_seed_all(seed)


# Define save_loss_plot; it receives the training history and the path where the figure should be saved.
def save_loss_plot(history, save_path):
    # Create a figure that is 10 inches wide and 6 inches high; figure is the standard function for starting a new plot.
    plt.figure(figsize=(10, 6))

    # The list comprehension [x[0] for x in ...] takes item 0 from every training record, which is the epoch number.
    train_x = [x[0] for x in history['train']]
    # In the same way, take item 1 from every training record, which is the average training loss.
    train_y = [x[1] for x in history['train']]
    # Plot the training-loss curve; label is used later by the legend to display Train Loss.
    plt.plot(train_x, train_y, label='Train Loss')

    # Draw the validation curve only after at least one validation result exists; an empty list cannot meaningfully represent a curve.
    if len(history['val']) > 0:
        # Take the epoch number from every validation record.
        val_x = [x[0] for x in history['val']]
        # Take the average validation loss from every validation record.
        val_y = [x[1] for x in history['val']]
        # Plot the validation-loss curve; marker='.' draws a small point at each validation result so the sparse validation positions are easy to see.
        plt.plot(val_x, val_y, label='Validation Loss', marker='.')

    # Set the x-axis label to Epoch so the plot clearly shows that the horizontal axis represents training rounds.
    plt.xlabel('Epoch')
    # Set the y-axis label to L1 Loss because the training objective in this project is the L1 noise-prediction loss.
    plt.ylabel('L1 Loss')
    # Set the title of the whole figure.
    plt.title('VoiceGrad Training Curve')
    # Turn on the grid; alpha=0.3 makes the grid light enough that it does not cover the main curves.
    plt.grid(True, alpha=0.3)
    # Automatically create the legend from the labels given in the plot calls.
    plt.legend()
    # Automatically adjust margins so the title and axis text are not cut off by the image boundary.
    plt.tight_layout()
    # Save the current figure to save_path; the caller decides the actual file path.
    plt.savefig(save_path)
    # Close the current figure and release memory; during long training, unclosed figures would keep accumulating.
    plt.close()


# Define train_one_epoch; this function is responsible only for training one complete epoch and then returning epoch-level statistics.
def train_one_epoch(model, diffusion, train_loader, optimizer, device):
    # model.train() switches the model to training mode; this is standard PyTorch syntax for telling the model that training is in progress.
    model.train()
    # Create an empty list to store the loss of every batch, so the average loss of the whole epoch can be calculated later.
    epoch_losses = []
    # Create an empty list to store the pre-clipping gradient norm of every batch for monitoring training stability.
    epoch_grad_norms = []

    # tqdm(train_loader,...) wraps the DataLoader with a progress bar; leave=False removes the bar after the epoch finishes.
    pbar = tqdm(train_loader, desc='Training', leave=False)

    # enumerate(pbar) returns both the batch index and the batch contents; batch_idx starts from 0 and increases by 1.
    for batch_idx, batch in enumerate(pbar):
        # batch['mel'] gets the mel tensor, and .to(device) moves it to the GPU or CPU; its shape is [B,80,T].
        mel = batch['mel'].to(device)
        # Get the 144-dimensional BNF aligned with the mel and move it to the same device; its shape is [B,144,T].
        bnf = batch['bnf'].to(device)
        # Get the target-speaker IDs and move them to the device; their shape is [B].
        spk = batch['spk_id'].to(device)

        # torch.randint randomly chooses one diffusion time step t for each speech sample in the batch.
        t = torch.randint(
            # low=0 means the smallest possible index is 0; the code uses Python's 0-based indexing.
            low=0,
            # high excludes the right boundary, so high=L produces values from 0 to L-1 and covers every diffusion level exactly.
            high=diffusion.n_levels,
            # size=(B,) means that each sample in the batch receives its own independently sampled t.
            size=(mel.shape[0],),
            # Create t directly on the training device to avoid errors caused by mixing CPU and GPU tensors.
            device=device
        # This closing parenthesis ends the torch.randint call.
        ).long()

        # torch.randn_like(mel) creates standard Gaussian noise epsilon~N(0,I) with exactly the same shape as mel.
        noise = torch.randn_like(mel)

        # Call the forward diffusion equation to add noise to clean mel x_0 and create x_t, which is the actual input seen by the model.
        x_t = diffusion.q_sample(mel, t, noise)

        # Call VoiceGrad and let it predict the noise that was just added, using x_t, the time step, the target speaker, and the BNF.
        pred_noise = model(x_t, t, spk, bnf)

        # F.l1_loss computes the mean absolute difference between predicted noise and true noise; the paper uses L1 instead of L2.
        loss = F.l1_loss(pred_noise, noise)

        # torch.isfinite checks whether the loss is a normal finite number; NaN or infinity means the training process has broken.
        if not torch.isfinite(loss):
            # Raise a RuntimeError immediately to stop training instead of continuing and saving a corrupted model.
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        # Clear gradients left by the previous batch; set_to_none=True uses less memory and is often faster than filling every gradient with zero.
        optimizer.zero_grad(set_to_none=True)
        # loss.backward() starts backpropagation from the loss and computes gradients for all trainable parameters.
        loss.backward()

        # clip_grad_norm_ computes the total gradient norm and scales it down when it is larger than the maximum value 1.0.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            # model.parameters() returns all trainable model parameters, and gradient clipping must consider them together.
            model.parameters(),
            # max_norm reads the largest allowed gradient norm from the configuration.
            max_norm=CONFIG['grad_clip_max_norm']
        # This closing parenthesis ends the gradient-clipping function call.
        )

        # Check whether the gradient norm is finite; if it is already NaN or Inf, updating parameters would be meaningless.
        if not torch.isfinite(grad_norm):
            # Stop immediately with an error when an invalid gradient is found.
            raise RuntimeError(f"Non-finite grad norm detected: {grad_norm}")

        # float(grad_norm) converts the one-element tensor into a normal Python float before storing it in the statistics list.
        epoch_grad_norms.append(float(grad_norm))

        # If the pre-clipping gradient is larger than the warning threshold, this batch may be unstable, so print a warning.
        if grad_norm > CONFIG['grad_warn_threshold']:
            # Print the batch index and pre-clipping gradient norm to help locate the unstable point.
            print(
                # The f-string inserts batch_idx into the warning message.
                f"[Grad Warning] batch={batch_idx} "
                # :.4f means display the gradient norm with four digits after the decimal point.
                f"pre_clip_grad_norm={float(grad_norm):.4f}"
            # This closing parenthesis ends the print call.
            )

        # optimizer.step() updates the model parameters using the gradients that were just computed; this is the step where the model actually learns.
        optimizer.step()

        # loss.item() converts the one-element loss tensor into a normal Python number before storing it in the epoch loss list.
        epoch_losses.append(loss.item())
        # set_postfix displays the current batch loss and gradient on the right side of the tqdm progress bar.
        pbar.set_postfix(
            # :.4f means display the loss with four digits after the decimal point.
            loss=f'{loss.item():.4f}',
            # :.3f means display the gradient with three digits after the decimal point.
            grad=f'{float(grad_norm):.3f}'
        # This closing parenthesis ends the set_postfix call.
        )

    # np.mean computes the average loss over all batches in this epoch, and float converts the result into a normal Python float.
    avg_loss = float(np.mean(epoch_losses))
    # Compute the average gradient norm over all batches in this epoch.
    avg_grad = float(np.mean(epoch_grad_norms))
    # np.max finds the largest gradient norm that appeared during this epoch.
    max_grad = float(np.max(epoch_grad_norms))

    # Return three values at once; Python packages them as a tuple that can be received by three variables outside the function.
    return avg_loss, avg_grad, max_grad


# @torch.no_grad() is a decorator; validation does not train the model, so gradient recording is disabled to save memory and computation.
@torch.no_grad()
# Define validate_one_epoch; it runs through the complete validation set without updating model parameters.
def validate_one_epoch(model, diffusion, val_loader, device):
    # model.eval() switches the model to evaluation mode; this is standard PyTorch practice before validation and inference.
    model.eval()
    # Create an empty list to store the loss of every validation sample or batch.
    val_losses = []

    # Wrap the validation DataLoader with a progress bar; leave=False removes it after validation finishes.
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    # Read validation data batch by batch; batch_idx is not needed here, so the loop simply uses for batch in pbar.
    for batch in pbar:
        # Get the validation mel and move it to the training device.
        mel = batch['mel'].to(device)
        # Get the validation BNF and move it to the same device.
        bnf = batch['bnf'].to(device)
        # Get the speaker ID of each validation sample and move it to the same device.
        spk = batch['spk_id'].to(device)

        # As in training, randomly choose one diffusion time step for each validation utterance.
        t = torch.randint(
            # The smallest index is 0.
            low=0,
            # diffusion.n_levels is excluded from the upper bound, so the actual range is 0 to L-1.
            high=diffusion.n_levels,
            # Sample one time step for each mel sample.
            size=(mel.shape[0],),
            # Create the tensor directly on the current device.
            device=device
        # This closing parenthesis ends the torch.randint call.
        ).long()

        # Generate standard Gaussian noise with the same shape as mel; this is the ground-truth noise target for validation.
        noise = torch.randn_like(mel)
        # Use the same forward diffusion equation to construct the noisy input x_t.
        x_t = diffusion.q_sample(mel, t, noise)
        # Ask the model to predict the noise; forward computation still happens, but no_grad prevents PyTorch from storing a backpropagation graph.
        pred_noise = model(x_t, t, spk, bnf)

        # Compute the L1 validation loss between the predicted noise and the true noise.
        loss = F.l1_loss(pred_noise, noise)

        # Check whether the validation loss contains NaN or infinity.
        if not torch.isfinite(loss):
            # Stop immediately if the loss is invalid, preventing a broken result from being treated as the best model.
            raise RuntimeError(f"Non-finite val loss detected: {loss.item()}")

        # Store the current validation loss as a normal number so the full validation average can be calculated later.
        val_losses.append(loss.item())

    # Return the average L1 loss over the entire validation set.
    return float(np.mean(val_losses))


# Define the main train function; it connects data loading, the model, diffusion, the optimizer, the training loop, and checkpoint saving.
def train():
    # Fix all random seeds first so the experiment is as reproducible as possible.
    set_seed(CONFIG['seed'])
    # os.makedirs creates the checkpoint directory; exist_ok=True prevents an error when the directory already exists.
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # Print the device actually being used so we can confirm whether the GPU was selected successfully.
    print(f"Device: {CONFIG['device']}")
    # Print a message showing that the program is starting to build the training set.
    print("Loading Training Set...")
    # Call get_dataloader from dataset.py to create the training DataLoader.
    train_loader = get_dataloader(
        # Pass the data root directory.
        CONFIG['data_root'],
        # split='train' tells the Dataset to filter files according to the paper's training split.
        split='train',
        # Read the training batch size from CONFIG.
        batch_size=CONFIG['batch_size'],
        # Read the number of parallel data-loading workers from CONFIG.
        num_workers=CONFIG['num_workers']
    # This closing parenthesis ends the get_dataloader call.
    )

    # Print a message showing that the program is starting to build the validation set.
    print("Loading Validation Set...")
    # Create the validation DataLoader.
    val_loader = get_dataloader(
        # Use the same data root directory.
        CONFIG['data_root'],
        # split='val' makes the Dataset keep only validation utterances 1001 to 1100.
        split='val',
        # Use batch_size=1 for validation because full-length utterances have different lengths and are difficult to combine into a larger batch.
        batch_size=1,
        # Use the same number of data-loading workers for validation.
        num_workers=CONFIG['num_workers']
    # This closing parenthesis ends the get_dataloader call.
    )

    # Create the main VoiceGrad model.
    model = VoiceGrad(
        # Fix the mel dimension at 80.
        n_mels=80,
        # Fix the BNF dimension at 144.
        n_bnf=144,
        # Set the main U-Net hidden width to 512 channels.
        n_channels=512,
        # There are four closed-set target speakers: clb, bdl, slt, and rms.
        n_spk=4
    # .to(device) moves all model parameters to the GPU or CPU.
    ).to(CONFIG['device'])

    # Create the diffusion-process object.
    diffusion = VoiceGradDiffusion(
        # The DPM setting in the paper uses L=20.
        n_levels=20,
        # Set eta=0.008 for the cosine schedule.
        offset=0.008
    # Move all registered diffusion buffers to the same device.
    ).to(CONFIG['device'])

    # Create the Adam optimizer; model.parameters() tells it which parameters to update, and lr sets the learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # Use a dictionary to record the complete training process so it can be plotted and saved inside checkpoints.
    history = {
        # train stores (epoch, train_loss).
        'train': [],
        # val stores (epoch, val_loss).
        'val': [],
        # grad_mean stores the average gradient norm of each epoch.
        'grad_mean': [],
        # grad_max stores the maximum gradient norm of each epoch.
        'grad_max': []
    # This closing brace ends the history dictionary.
    }
    # float('inf') means positive infinity; the first validation loss will always be smaller, so the first best checkpoint is guaranteed to be saved.
    best_val_loss = float('inf')

    # Print the total number of training epochs to confirm the configuration.
    print(f"Start Training for {CONFIG['epochs']} epochs...")

    # Use try so a Ctrl+C KeyboardInterrupt can be caught and an emergency checkpoint can be saved before the program exits.
    try:
        # range(CONFIG['epochs']) produces 0 to epochs-1, and each loop iteration trains one epoch.
        for epoch in range(CONFIG['epochs']):
            # Epoch numbers starting from 0 are inconvenient for people, so add 1 to obtain the familiar range 1 to 4000.
            epoch_num = epoch + 1

            # Call train_one_epoch to train one complete epoch, then receive its three returned statistics in three separate variables.
            avg_train_loss, avg_grad, max_grad = train_one_epoch(
                # Pass the current model.
                model=model,
                # Pass the diffusion process.
                diffusion=diffusion,
                # Pass the training DataLoader.
                train_loader=train_loader,
                # Pass the optimizer.
                optimizer=optimizer,
                # Pass the device.
                device=CONFIG['device']
            # This closing parenthesis ends the train_one_epoch call.
            )

            # append adds the current epoch and average training loss as a tuple to the history list.
            history['train'].append((epoch_num, avg_train_loss))
            # Store the average gradient norm for the current epoch.
            history['grad_mean'].append((epoch_num, avg_grad))
            # Store the maximum gradient norm for the current epoch.
            history['grad_max'].append((epoch_num, max_grad))

            # Build one training log message from several adjacent f-strings; Python automatically joins adjacent strings inside parentheses.
            log_msg = (
                # :04d displays the epoch as a four-digit integer, so 1 appears as 0001.
                f"Epoch {epoch_num:04d} | "
                # :.4f displays the training loss with four digits after the decimal point.
                f"Train: {avg_train_loss:.4f} | "
                # Display the average gradient norm.
                f"GradMean: {avg_grad:.4f} | "
                # Display the maximum gradient norm.
                f"GradMax: {max_grad:.4f}"
            # This closing parenthesis ends the string construction.
            )

            # % is the remainder operator; when epoch_num is divisible by val_every, the remainder is 0, so validation runs every 20 epochs.
            if epoch_num % CONFIG['val_every'] == 0:
                # Call the validation function and receive the average validation loss.
                avg_val_loss = validate_one_epoch(
                    # Pass the current model.
                    model=model,
                    # Pass the same diffusion object.
                    diffusion=diffusion,
                    # Pass the validation DataLoader.
                    val_loader=val_loader,
                    # Pass the device.
                    device=CONFIG['device']
                # This closing parenthesis ends the validation-function call.
                )
                # Store the current validation epoch and loss in history.
                history['val'].append((epoch_num, avg_val_loss))
                # += appends new text to the existing string; here it adds the validation loss to the log message.
                log_msg += f" | Val: {avg_val_loss:.4f}"

                # If the current validation loss is lower than the best value seen so far, update the best model.
                if avg_val_loss < best_val_loss:
                    # First store the new best validation loss for future comparisons.
                    best_val_loss = avg_val_loss
                    # Create the best-checkpoint dictionary and save all information needed to resume training.
                    best_ckpt = {
                        # Save the current epoch number.
                        'epoch': epoch_num,
                        # state_dict() returns the names and values of all trainable model parameters.
                        'model': model.state_dict(),
                        # Save the optimizer state so future training can resume with the same Adam momentum values.
                        'optimizer': optimizer.state_dict(),
                        # Save the current best validation loss.
                        'best_val_loss': best_val_loss,
                        # Save the complete training history.
                        'history': history,
                        # Save the training configuration so we know how this model was trained in the future.
                        'config': CONFIG
                    # This closing brace ends the best_ckpt dictionary.
                    }
                    # torch.save serializes the checkpoint and writes it to best_model.pt.
                    torch.save(best_ckpt, os.path.join(CONFIG['save_dir'], 'best_model.pt'))
                    # Append Best Updated to the log so it is clear that this epoch produced a new best validation result.
                    log_msg += " | [Best Updated]"

            # Print the final log for this epoch; if no validation ran, it contains only training and gradient information.
            print(log_msg)

            # Create a latest checkpoint every epoch, whether or not it is the best one, so an unexpected disconnection loses at most one epoch.
            latest_ckpt = {
                # Save the current epoch.
                'epoch': epoch_num,
                # Save the current model parameters.
                'model': model.state_dict(),
                # Save the current optimizer state.
                'optimizer': optimizer.state_dict(),
                # Save the best validation loss seen so far.
                'best_val_loss': best_val_loss,
                # Save the complete history.
                'history': history,
                # Save the configuration.
                'config': CONFIG
            # This closing brace ends the latest_ckpt dictionary.
            }
            # Overwrite latest_model.pt so it always represents the newest training state.
            torch.save(latest_ckpt, os.path.join(CONFIG['save_dir'], 'latest_model.pt'))

            # Whenever epoch_num is divisible by save_every, save an additional historical checkpoint that will not be overwritten.
            if epoch_num % CONFIG['save_every'] == 0:
                # os.path.join safely combines the directory and filename, and the f-string inserts the epoch number into the filename.
                ckpt_path = os.path.join(CONFIG['save_dir'], f'model_epoch_{epoch_num}.pt')
                # Save the current latest_ckpt contents to the permanent historical checkpoint file.
                torch.save(latest_ckpt, ckpt_path)
                # Print the actual save path so we can confirm that the checkpoint was written successfully.
                print(f"Checkpoint saved: {ckpt_path}")

            # Whenever epoch_num is divisible by plot_every, update the training curve and the history data file.
            if epoch_num % CONFIG['plot_every'] == 0:
                # Call the previously defined save_loss_plot function to save loss_curve.png.
                save_loss_plot(
                    # Pass the complete history.
                    history,
                    # Build the image save path.
                    os.path.join(CONFIG['save_dir'], 'loss_curve.png')
                # This closing parenthesis ends the save_loss_plot call.
                )
                # np.save stores the history dictionary in an .npy file so curves can later be analyzed without opening a checkpoint.
                np.save(
                    # The first argument is the save path.
                    os.path.join(CONFIG['save_dir'], 'loss_history.npy'),
                    # The second argument is the history object to save.
                    history,
                    # allow_pickle=True allows NumPy to save a Python dictionary, which is not a normal numeric array.
                    allow_pickle=True
                # This closing parenthesis ends the np.save call.
                )

    # except KeyboardInterrupt catches Ctrl+C so the program can save before exiting when the user stops training manually.
    except KeyboardInterrupt:
        # Print a message showing that an emergency checkpoint is being saved.
        print("\n[Interrupted] Saving emergency checkpoint...")
        # Create the interruption checkpoint.
        interrupt_ckpt = {
            # Conditional expression: save epoch_num if it already exists; otherwise training has not started yet, so save 0.
            'epoch': epoch_num if 'epoch_num' in locals() else 0,
            # Save the model parameters at the moment of interruption.
            'model': model.state_dict(),
            # Save the optimizer state at the moment of interruption.
            'optimizer': optimizer.state_dict(),
            # Save the current best validation loss.
            'best_val_loss': best_val_loss,
            # Save the training history collected so far.
            'history': history,
            # Save the configuration.
            'config': CONFIG
        # This closing brace ends the interrupt_ckpt dictionary.
        }
        # Save the emergency checkpoint as interrupt_model.pt.
        torch.save(interrupt_ckpt, os.path.join(CONFIG['save_dir'], 'interrupt_model.pt'))
        # Print a message confirming that the emergency checkpoint was saved.
        print("Emergency checkpoint saved.")
        # A bare raise re-raises the KeyboardInterrupt that was just caught so the program actually terminates.
        raise

    # Print a completion message after all epochs finish normally.
    print("Training finished.")


# Enter train() only when python train_practice.py is executed directly; importing the file from another module does not start training.
if __name__ == '__main__':
    # Call the main training entry point and begin training.
    train()
