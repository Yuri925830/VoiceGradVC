import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import VoiceGrad
from diffusion import VoiceGradDiffusion
from dataset import get_dataloader


CONFIG = {
    # root folder of dataset containing the mel / bnf / stats sub-folder
    'data_root': '/content/drive/MyDrive/VoiceGrad/data',

    # training settings
    'epochs': 4000, 
    'batch_size': 16, 
    'lr': 1e-4, 
    'num_workers': 4, 
    'seed': 1234, 

    # gradient clipping
    'grad_clip_max_norm': 1.0,
    'grad_warn_threshold': 5.0,

    # validation / saving schedules
    'val_every': 20, 
    'plot_every': 50, 
    'save_every': 500, 

    # paths
    'save_dir': './checkpoints', 

    # device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_loss_plot(history, save_path):
    plt.figure(figsize=(10, 6))

    train_x = [x[0] for x in history['train']]
    train_y = [x[1] for x in history['train']]
    plt.plot(train_x, train_y, label='Train Loss')

    if len(history['val']) > 0:
        val_x = [x[0] for x in history['val']]
        val_y = [x[1] for x in history['val']]
        plt.plot(val_x, val_y, label='Validation Loss', marker='.')

    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title('VoiceGrad Training Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_one_epoch(model, diffusion, train_loader, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_grad_norms = []

    pbar = tqdm(train_loader, desc='Training', leave=False)

    for batch_idx, batch in enumerate(pbar):
        mel = batch['mel'].to(device)          # [B, 80, T]
        bnf = batch['bnf'].to(device)          # [B, 144, T]
        spk = batch['spk_id'].to(device)       # [B]

        # pick a random diffusion step 't' for each sample in the batch
        # in the math this is the noise level 1 ~ Uniform(1, ...,L)
        # the code uses 0-based indexing, so the range is 0 ... L-1
        t = torch.randint(
            low=0,
            high=diffusion.n_levels,
            size=(mel.shape[0],),
            device=device
        ).long()

        # epsilon ~ N(0, I)
        noise = torch.randn_like(mel)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = diffusion.q_sample(mel, t, noise)

        # epsilon_theta(x_t, t, k, bnf)
        pred_noise = model(x_t, t, spk, bnf)

        # L1 loss
        loss = F.l1_loss(pred_noise, noise)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping. rescales the gradients so their combined size never exceeds grad_clip_max_norm
        # return value is the total gradient norm measured before clipping which is logged below
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CONFIG['grad_clip_max_norm']
        )

        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite grad norm detected: {grad_norm}")

        epoch_grad_norms.append(float(grad_norm))

        # if the pre-clipped gradient was unusally large, print warning can find which batch it's from
        # clipping still keeps training stable but frequent warnings hint something may be off
        if grad_norm > CONFIG['grad_warn_threshold']:
            print(
                f"[Grad Warning] batch={batch_idx} "
                f"pre_clip_grad_norm={float(grad_norm):.4f}"
            )

        optimizer.step()

        epoch_losses.append(loss.item())
        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            grad=f'{float(grad_norm):.3f}'
        )

    avg_loss = float(np.mean(epoch_losses))
    avg_grad = float(np.mean(epoch_grad_norms))
    max_grad = float(np.max(epoch_grad_norms))

    return avg_loss, avg_grad, max_grad


@torch.no_grad()
def validate_one_epoch(model, diffusion, val_loader, device):
    model.eval()
    val_losses = []

    pbar = tqdm(val_loader, desc='Validation', leave=False)
    for batch in pbar:
        mel = batch['mel'].to(device)
        bnf = batch['bnf'].to(device)
        spk = batch['spk_id'].to(device)

        t = torch.randint(
            low=0,
            high=diffusion.n_levels,
            size=(mel.shape[0],),
            device=device
        ).long()

        noise = torch.randn_like(mel)
        x_t = diffusion.q_sample(mel, t, noise)
        pred_noise = model(x_t, t, spk, bnf)

        loss = F.l1_loss(pred_noise, noise)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite val loss detected: {loss.item()}")

        val_losses.append(loss.item())

    return float(np.mean(val_losses))


def train():
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    print(f"Device: {CONFIG['device']}")
    print("Loading Training Set...")
    train_loader = get_dataloader(
        CONFIG['data_root'],
        split='train',
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    print("Loading Validation Set...")
    val_loader = get_dataloader(
        CONFIG['data_root'],
        split='val',
        batch_size=1,
        num_workers=CONFIG['num_workers']
    )

    # K = 4（clb, bdl, slt, rms）
    model = VoiceGrad(
        n_mels=80,
        n_bnf=144,
        n_channels=512,
        n_spk=4
    ).to(CONFIG['device'])

    diffusion = VoiceGradDiffusion(
        n_levels=20,
        offset=0.008
    ).to(CONFIG['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    history = {
        'train': [],
        'val': [],
        'grad_mean': [],
        'grad_max': []
    }
    best_val_loss = float('inf')

    print(f"Start Training for {CONFIG['epochs']} epochs...")

    try:
        for epoch in range(CONFIG['epochs']):
            epoch_num = epoch + 1

            avg_train_loss, avg_grad, max_grad = train_one_epoch(
                model=model,
                diffusion=diffusion,
                train_loader=train_loader,
                optimizer=optimizer,
                device=CONFIG['device']
            )

            history['train'].append((epoch_num, avg_train_loss))
            history['grad_mean'].append((epoch_num, avg_grad))
            history['grad_max'].append((epoch_num, max_grad))

            log_msg = (
                f"Epoch {epoch_num:04d} | "
                f"Train: {avg_train_loss:.4f} | "
                f"GradMean: {avg_grad:.4f} | "
                f"GradMax: {max_grad:.4f}"
            )

            # validation
            if epoch_num % CONFIG['val_every'] == 0:
                avg_val_loss = validate_one_epoch(
                    model=model,
                    diffusion=diffusion,
                    val_loader=val_loader,
                    device=CONFIG['device']
                )
                history['val'].append((epoch_num, avg_val_loss))
                log_msg += f" | Val: {avg_val_loss:.4f}"

                # saving the best model whenever validation loss improves
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt = {
                        'epoch': epoch_num,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'history': history,
                        'config': CONFIG
                    }
                    torch.save(best_ckpt, os.path.join(CONFIG['save_dir'], 'best_model.pt'))
                    log_msg += " | [Best Updated]"

            print(log_msg)

            # always save the latest model after every epoch
            # this is the 'resume from here if the run dies' checkpoint
            latest_ckpt = {
                'epoch': epoch_num,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': CONFIG
            }
            torch.save(latest_ckpt, os.path.join(CONFIG['save_dir'], 'latest_model.pt'))

            # periodically save a numbered historical checkpoint
            # keeps snapshots at fixed epochs so versions can be compared later
            if epoch_num % CONFIG['save_every'] == 0:
                ckpt_path = os.path.join(CONFIG['save_dir'], f'model_epoch_{epoch_num}.pt')
                torch.save(latest_ckpt, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            # periodically draw and save the loss curve
            if epoch_num % CONFIG['plot_every'] == 0:
                save_loss_plot(
                    history,
                    os.path.join(CONFIG['save_dir'], 'loss_curve.png')
                )
                np.save(
                    os.path.join(CONFIG['save_dir'], 'loss_history.npy'),
                    history,
                    allow_pickle=True
                )

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving emergency checkpoint...")
        interrupt_ckpt = {
            'epoch': epoch_num if 'epoch_num' in locals() else 0,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            'config': CONFIG
        }
        torch.save(interrupt_ckpt, os.path.join(CONFIG['save_dir'], 'interrupt_model.pt'))
        print("Emergency checkpoint saved.")
        raise

    print("Training finished.")


if __name__ == '__main__':
    train()
