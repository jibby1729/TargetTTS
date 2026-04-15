import torch
import torch.nn as nn
import pandas as pd
from torch.amp import autocast
from torch.utils.data import DataLoader
from src.extraction_net import ExtractionNet
from src.dataset import MixtureDataset, pad_collate

def power_loss(estimated, target, power):
    eps = 1e-8
    return nn.functional.mse_loss((estimated + eps).pow(power), (target + eps).pow(power))

device = "cuda"
power = 0.3

df_val = pd.read_csv("data/prepared_training/val.csv")
embeddings = torch.load("data/prepared_training/embeddings.pt", weights_only=False)
val_loader = DataLoader(
    MixtureDataset(df_val, "data/synthetic_mixtures/librispeech_extraction", embeddings, max_frames=1400),
    batch_size=16, shuffle=False, collate_fn=pad_collate, num_workers=0,
)

checkpoints = {
    "hybrid (α=0.1)":  ("checkpoints/hybrid_alpha_0.1/best_model.pt",  0.1),
    "power_law_only":  ("checkpoints/power_law_only/best_model.pt",     None),
    "curriculum_easy": ("checkpoints/curriculum_easy/best_model.pt",    0.1),
    "hard_power_law":  ("checkpoints/hard_power_law/best_model.pt",     None),
}

print(f"{'Model':<22}  {'Val Loss (train obj)':>20}  {'Val Recon PL':>14}  {'Val Mask MSE':>14}")
print("-" * 80)

for name, (path, alpha) in checkpoints.items():
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ExtractionNet(input_power=power).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_loss = total_recon = total_mask = n = 0
    with torch.no_grad():
        for mix_mag, target_mag, speaker_emb in val_loader:
            mix_mag, target_mag, speaker_emb = mix_mag.to(device), target_mag.to(device), speaker_emb.to(device)
            with autocast(device_type=device, dtype=torch.bfloat16):
                mask = model(mix_mag, speaker_emb)
            recon = power_loss(mask * mix_mag, target_mag, power).item()
            oracle = torch.clamp(target_mag / (mix_mag + 1e-8), 0, 1)
            mask_mse = nn.functional.mse_loss(mask, oracle).item()
            loss = (alpha * mask_mse + (1 - alpha) * recon) if alpha is not None else recon
            total_loss += loss; total_recon += recon; total_mask += mask_mse; n += 1

    print(f"{name:<22}  {total_loss/n:>20.6f}  {total_recon/n:>14.6f}  {total_mask/n:>14.6f}")
