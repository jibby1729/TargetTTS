"""
Speaker embedding extraction using a pretrained ECAPA-TDNN model.

We use SpeechBrain's ECAPA-TDNN checkpoint (trained on VoxCeleb1+2) to convert
a short audio clip of a speaker into a 192-dimensional embedding vector. This
embedding captures the speaker's identity and is used to condition the extraction
network so it knows which speaker to isolate from a mixture.

The model is kept frozen throughout our pipeline, we never fine-tune it.

Example:
    encoder = SpeakerEncoder()

    # From a file path
    embedding = encoder.encode_file("path/to/enrollment.flac")
    print(embedding.shape)  # torch.Size([192])

    # From a waveform tensor
    waveform = torch.randn(16000 * 3)  # 3 seconds of audio
    embedding = encoder.encode_waveform(waveform)
"""

import torch
from speechbrain.inference.speaker import EncoderClassifier

from src.preprocess import load_audio, TARGET_SAMPLE_RATE


class SpeakerEncoder:
    """
    Wraps SpeechBrain's pretrained ECAPA-TDNN to extract fixed-size speaker embeddings.

    The model downloads automatically on first use and is cached locally.
    All parameters are frozen — this module is inference-only.
    """

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load the pretrained ECAPA-TDNN model from SpeechBrain's HuggingFace hub.
        # This produces 192-dimensional embeddings.
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

        # Make sure nothing gets updated during training
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_file(self, audio_path):
        """
        Load an audio file and return its speaker embedding.

        Uses our existing load_audio function (soundfile-based, no FFmpeg needed).
        Expects 16kHz mono audio (which all our LibriSpeech data already is).

        Args:
            audio_path: Path to an audio file (.flac, .wav, etc.)

        Returns:
            A 192-dimensional tensor representing the speaker's voice.
        """
        # load_audio returns shape (1, num_samples) — already mono and 16kHz
        signal = load_audio(audio_path)

        # SpeechBrain expects shape (batch, time), so (1, num_samples) works directly
        signal = signal.to(self.device)
        embedding = self.model.encode_batch(signal)  # returns (1, 1, 192)

        return embedding.squeeze().cpu()  # (192,)

    @torch.no_grad()
    def encode_waveform(self, waveform):
        """
        Extract a speaker embedding from a raw waveform tensor.

        This is useful when we already have the audio loaded in memory
        (e.g., from our preprocessing pipeline) and don't want to re-read from disk.

        Args:
            waveform: A 1D tensor of shape (num_samples,) at 16kHz.

        Returns:
            A 192-dimensional tensor representing the speaker's voice.
        """
        # Add a batch dimension if needed: (num_samples,) -> (1, num_samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)
        embedding = self.model.encode_batch(waveform)  # (1, 1, 192)

        return embedding.squeeze().cpu()  # (192,)
