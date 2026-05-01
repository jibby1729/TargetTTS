"""
PyTorch dataset and collation for mixture/target spectrogram pairs.

Used by both the training script and the evaluation script. Each sample
loads a mixture and its aligned clean target, computes their magnitude
spectrograms, and looks up the pre-computed speaker embedding.

Example:
    dataset = MixtureDataset(df, mix_dir, embeddings)
    loader = DataLoader(dataset, batch_size=8, collate_fn=pad_collate)

    for mix_mag, target_mag, speaker_emb in loader:
        # mix_mag:     (B, 257, T)
        # target_mag:  (B, 257, T)
        # speaker_emb: (B, 192)
        ...
"""

import os
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.extraction_net import compute_stft


class MixtureDataset(Dataset):
    """
    Loads mixture/target audio pairs and looks up pre-computed speaker embeddings.

    Each item returns:
        mix_mag:     (F, T) magnitude spectrogram of the mixture
        target_mag:  (F, T) magnitude spectrogram of the clean aligned target
        speaker_emb: (192,) pre-computed speaker embedding
    """

    def __init__(self, recipes_df, mix_dir, embeddings, max_frames=None):
        self.recipes = recipes_df.reset_index(drop=True)
        self.mix_dir = mix_dir
        self.embeddings = embeddings
        self.max_frames = max_frames

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx):
        row = self.recipes.iloc[idx]
        mix_id = row["mix_id"]

        # Load the mixture and clean target audio
        mix_wav, _ = sf.read(
            os.path.join(self.mix_dir, f"{mix_id}.wav"), dtype="float32"
        )
        target_wav, _ = sf.read(
            os.path.join(self.mix_dir, f"{mix_id}_target.wav"), dtype="float32"
        )

        # Compute magnitude spectrograms
        mix_mag, _ = compute_stft(torch.from_numpy(mix_wav))       # (1, F, T)
        target_mag, _ = compute_stft(torch.from_numpy(target_wav))  # (1, F, T)

        # Remove the batch dimension that compute_stft adds
        mix_mag = mix_mag.squeeze(0)       # (F, T)
        target_mag = target_mag.squeeze(0)  # (F, T)

        # Random crop to max_frames (prevents OOM from very long utterances)
        if self.max_frames is not None and mix_mag.shape[1] > self.max_frames:
            start = random.randint(0, mix_mag.shape[1] - self.max_frames)
            mix_mag = mix_mag[:, start:start + self.max_frames]
            target_mag = target_mag[:, start:start + self.max_frames]

        # Look up the pre-computed speaker embedding
        speaker_emb = self.embeddings[row["enrollment_audio_path"]]

        return mix_mag, target_mag, speaker_emb


def pad_collate(batch):
    """
    Pads spectrograms to the longest sequence in the batch.

    Different utterances have different lengths, so we zero-pad shorter
    ones on the right to form a uniform batch.
    """
    mix_mags, target_mags, speaker_embs = zip(*batch)

    max_t = max(m.shape[1] for m in mix_mags)

    padded_mix = []
    padded_target = []
    for mix_m, tgt_m in zip(mix_mags, target_mags):
        pad_amount = max_t - mix_m.shape[1]
        if pad_amount > 0:
            mix_m = torch.nn.functional.pad(mix_m, (0, pad_amount))
            tgt_m = torch.nn.functional.pad(tgt_m, (0, pad_amount))
        padded_mix.append(mix_m)
        padded_target.append(tgt_m)

    return (
        torch.stack(padded_mix),       # (B, F, T)
        torch.stack(padded_target),    # (B, F, T)
        torch.stack(speaker_embs),     # (B, 192)
    )
