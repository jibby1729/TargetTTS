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


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm for Conv1D output where the channel dimension is not last.

    Conv1D produces (B, C, T), but nn.LayerNorm normalizes the last dimension.
    This wrapper transposes to (B, T, C), applies LayerNorm, and transposes back.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) -> LayerNorm -> (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ExtractionNet(nn.Module):
    """
    Predicts a time-frequency mask to extract a target speaker from a mixture,
    conditioned on a speaker embedding.

    Input:
        magnitude_spec: (B, F, T) magnitude spectrogram of the mixture
        speaker_emb:    (B, D) speaker embedding from ECAPA-TDNN

    Output:
        mask: (B, F, T) soft mask with values in [0, 1]
    """

    def __init__(self, n_freq=STFT_N_FREQ, embedding_dim=192, conv_channels=256,
                 n_conv_layers=5, lstm_hidden=400):
        super().__init__()

        self.n_freq = n_freq
        self.embedding_dim = embedding_dim

        # --- Convolutional encoder ---
        # 5 dilated Conv1D layers that process the spectrogram along the time axis.
        # Dilation doubles each layer: 1, 2, 4, 8, 16
        # This gives a receptive field of 31 frames (~310ms at 10ms hop).
        conv_layers = []
        in_channels = n_freq
        for i in range(n_conv_layers):
            dilation = 2 ** i
            padding = dilation  # "same" padding for kernel size 3
            conv_layers.append(
                nn.Conv1d(in_channels, conv_channels, kernel_size=3,
                          dilation=dilation, padding=padding)
            )
            conv_layers.append(ChannelLayerNorm(conv_channels))
            conv_layers.append(nn.ReLU())
            in_channels = conv_channels

        self.conv_encoder = nn.Sequential(*conv_layers)

        # --- Bidirectional LSTM ---
        # Takes the CNN features concatenated with the speaker embedding.
        # The BiLSTM sees the full utterance in both directions, which helps
        # resolve ambiguous frames where both speakers overlap.
        lstm_input_size = conv_channels + embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Mask predictor ---
        # Two FC layers: the first with ReLU, the second with sigmoid
        # to produce mask values in [0, 1] for each frequency bin.
        self.fc1 = nn.Linear(lstm_hidden * 2, n_freq)
        self.fc2 = nn.Linear(n_freq, n_freq)

    def forward(self, magnitude_spec, speaker_emb):
        """
        Args:
            magnitude_spec: (B, F, T) mixture magnitude spectrogram
            speaker_emb:    (B, D) speaker embedding from enrollment clip

        Returns:
            mask: (B, F, T) soft mask, values in [0, 1]
        """
        batch_size, F, T = magnitude_spec.shape

        # CNN expects (B, channels, time) — our spectrogram is already (B, F, T)
        # where F acts as the channel dimension and T is the time axis.
        cnn_out = self.conv_encoder(magnitude_spec)  # (B, C, T)

        # Transpose to (B, T, C) for the LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (B, T, C)

        # Concatenate speaker embedding at every time frame.
        # Expand (B, D) to (B, T, D) so we can concatenate along the feature axis.
        emb_expanded = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        lstm_input = torch.cat([cnn_out, emb_expanded], dim=2)  # (B, T, C+D)

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
