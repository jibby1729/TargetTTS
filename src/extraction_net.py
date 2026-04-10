"""
Speaker-conditioned mask network for target speaker extraction.

This is the only trained component in our pipeline. It takes the magnitude
spectrogram of a mixture and a speaker embedding from ECAPA-TDNN, and predicts
a soft mask that isolates the target speaker's signal.

Architecture:
  - 5 dilated Conv1D layers with LayerNorm + ReLU
  - Speaker embedding concatenated to the CNN output at every frame
  - Bidirectional LSTM for temporal modelling
  - Two FC layers ending in sigmoid to produce the mask

The mask is applied to the mixture spectrogram, and the result is inverted
back to a waveform using the original mixture phase.

Example:
    model = ExtractionNet()

    # magnitude_spec: (batch, freq_bins, time_frames) — e.g. (B, 257, T)
    # speaker_emb:    (batch, 192)
    mask = model(magnitude_spec, speaker_emb)  # (B, 257, T), values in [0, 1]
"""

import torch
import torch.nn as nn

from src.preprocess import TARGET_SAMPLE_RATE

# STFT parameters — standard for 16kHz speech
STFT_WIN_LENGTH = 400   # 25ms window
STFT_HOP_LENGTH = 160   # 10ms hop
STFT_N_FFT = 512        # FFT size
STFT_N_FREQ = STFT_N_FFT // 2 + 1  # 257 frequency bins


class Conv2dLayerNorm(nn.Module):
    """
    LayerNorm-equivalent for Conv2D output using GroupNorm(1).

    Conv2D produces (B, C, F, T). GroupNorm with num_groups=1 normalizes
    over all of (C, F, T) per batch element — equivalent to LayerNorm
    but operates natively on 4D tensors without expensive permutations.
    """

    def __init__(self, n_channels, n_freq=None):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_channels)

    def forward(self, x):
        return self.norm(x)


class ExtractionNet(nn.Module):
    """
    Predicts a time-frequency mask to extract a target speaker from a mixture,
    conditioned on a speaker embedding.

    Uses 2D convolutions following the VoiceFilter architecture (Wang et al.,
    2019), operating on the spectrogram as a 2D time-frequency image. The CNN
    learns local time-frequency patterns (harmonics, formants) with weight
    sharing across frequency, then a BiLSTM integrates temporal context.

    Input:
        magnitude_spec: (B, F, T) magnitude spectrogram of the mixture
        speaker_emb:    (B, D) speaker embedding from ECAPA-TDNN

    Output:
        mask: (B, F, T) soft mask with values in [0, 1]
    """

    def __init__(self, n_freq=STFT_N_FREQ, embedding_dim=192, conv_channels=64,
                 bottleneck_channels=8, lstm_hidden=400, input_power=None):
        super().__init__()

        self.n_freq = n_freq
        self.embedding_dim = embedding_dim
        self.input_power = input_power
        self.bottleneck_channels = bottleneck_channels

        # --- 2D Convolutional encoder (VoiceFilter architecture) ---
        # 8 Conv2D layers operating on (B, C, F, T).
        # Layers 1-2: separable-style (frequency-only, then time-only)
        # Layers 3-7: joint time-frequency with increasing dilation in time
        # Layer 8: 1x1 bottleneck to reduce channels before the LSTM
        #
        # Dilation is only applied along the time axis (never frequency),
        # giving a temporal receptive field of 31 frames (~310ms).
        # Frequency receptive field stays local (~29 bins ≈ 900 Hz).
        self.conv_encoder = nn.Sequential(
            # Layer 1: frequency-only (7, 1) — learns local spectral patterns
            nn.Conv2d(1, conv_channels, kernel_size=(7, 1), padding=(3, 0)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 2: time-only (1, 7) — learns temporal dynamics
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 7), padding=(0, 3)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 3: joint (5, 5), no dilation
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5),
                      dilation=(1, 1), padding=(2, 2)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 4: joint (5, 5), dilation 2 in time only
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5),
                      dilation=(1, 2), padding=(2, 4)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 5: joint (5, 5), dilation 4 in time only
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5),
                      dilation=(1, 4), padding=(2, 8)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 6: joint (5, 5), dilation 8 in time only
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5),
                      dilation=(1, 8), padding=(2, 16)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 7: joint (5, 5), dilation 16 in time only
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(5, 5),
                      dilation=(1, 16), padding=(2, 32)),
            Conv2dLayerNorm(conv_channels, n_freq),
            nn.ReLU(),
            # Layer 8: 1x1 bottleneck to reduce channels
            nn.Conv2d(conv_channels, bottleneck_channels, kernel_size=(1, 1)),
            Conv2dLayerNorm(bottleneck_channels, n_freq),
            nn.ReLU(),
        )

        # --- Bidirectional LSTM ---
        # Input: flattened CNN output (bottleneck_channels * n_freq) + speaker embedding
        lstm_input_size = bottleneck_channels * n_freq + embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Mask predictor (VoiceFilter-style) ---
        # Two FC layers with a wide intermediate (600) to avoid bottlenecking
        # the per-frequency mask prediction.
        self.fc1 = nn.Linear(lstm_hidden * 2, 600)
        self.fc2 = nn.Linear(600, n_freq)

    def forward(self, magnitude_spec, speaker_emb):
        """
        Args:
            magnitude_spec: (B, F, T) mixture magnitude spectrogram
            speaker_emb:    (B, D) speaker embedding from enrollment clip

        Returns:
            mask: (B, F, T) soft mask, values in [0, 1]
        """
        batch_size, F, T = magnitude_spec.shape

        # Optionally compress the input for the CNN
        cnn_input = magnitude_spec
        if self.input_power is not None:
            cnn_input = (magnitude_spec + 1e-8).pow(self.input_power)

        # Add a channel dimension: (B, F, T) -> (B, 1, F, T)
        # The 2D CNN treats the spectrogram as a single-channel image
        # with frequency as height and time as width.
        cnn_input = cnn_input.unsqueeze(1)  # (B, 1, F, T)

        cnn_out = self.conv_encoder(cnn_input)  # (B, bottleneck_channels, F, T)

        # Flatten channel and frequency dims, then transpose to (B, T, features)
        # (B, bottleneck_channels, F, T) -> (B, bottleneck_channels * F, T) -> (B, T, bottleneck_channels * F)
        cnn_out = cnn_out.reshape(batch_size, self.bottleneck_channels * F, T)
        cnn_out = cnn_out.transpose(1, 2)  # (B, T, bottleneck_channels * F)

        # Concatenate speaker embedding at every time frame
        emb_expanded = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        lstm_input = torch.cat([cnn_out, emb_expanded], dim=2)  # (B, T, 256 + D)

        # BiLSTM processes the full sequence in both directions
        lstm_out, _ = self.lstm(lstm_input)  # (B, T, 2*H)

        # Predict mask values at each time-frequency bin
        z = torch.relu(self.fc1(lstm_out))    # (B, T, F)
        mask = torch.sigmoid(self.fc2(z))     # (B, T, F)

        # Transpose back to (B, F, T) to match the spectrogram layout
        mask = mask.transpose(1, 2)  # (B, F, T)

        return mask


def compute_stft(waveform):
    """
    Compute the STFT of a waveform, returning magnitude and phase separately.

    We need the full linear-frequency magnitude (not log-mel) because we have to
    invert back to a waveform after masking. The phase is kept for reconstruction.

    Args:
        waveform: (B, num_samples) or (num_samples,) tensor

    Returns:
        magnitude: (B, F, T) magnitude spectrogram
        phase:     (B, F, T) phase angles
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(STFT_WIN_LENGTH, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        win_length=STFT_WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    # stft shape: (B, F, T)
    magnitude = stft.abs()
    phase = stft.angle()
    return magnitude, phase


def apply_mask_and_istft(magnitude, phase, mask):
    """
    Apply a mask to a magnitude spectrogram and reconstruct the waveform.

    The masked magnitude is combined with the original mixture phase to form
    a complex spectrogram, which is then inverted back to a time-domain signal.

    Args:
        magnitude: (B, F, T) original mixture magnitude
        phase:     (B, F, T) original mixture phase
        mask:      (B, F, T) predicted mask from the extraction network

    Returns:
        waveform: (B, num_samples) reconstructed time-domain signal
    """
    masked_magnitude = mask * magnitude
    complex_spec = torch.polar(masked_magnitude, phase)

    window = torch.hann_window(STFT_WIN_LENGTH, device=magnitude.device)
    waveform = torch.istft(
        complex_spec,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP_LENGTH,
        win_length=STFT_WIN_LENGTH,
        window=window,
    )
    return waveform
