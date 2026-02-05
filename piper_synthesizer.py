"""
Piper TTS Synthesizer Wrapper.

Handles synthesis with Piper TTS and generates audio from personalized profiles.
"""

import logging
import struct
import io
import time
import psutil
import wave
from pathlib import Path
from typing import Optional, Tuple
from piper.voice import PiperVoice, SynthesisConfig

logger = logging.getLogger(__name__)
process = psutil.Process()

# Try to import Piper
try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
    logger.info("Piper TTS library available")
except ImportError:
    PIPER_AVAILABLE = False
    logger.warning("Piper TTS not installed. Install with: pip install piper-tts")


def _log_perf(stage_name: str, start_time: float) -> None:
    """Log performance metrics: duration, CPU%, memory%."""
    duration = time.perf_counter() - start_time
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    logger.info(
        "PERF | %-20s | time=%.3fs | cpu=%.1f%% | mem=%.1f%%",
        stage_name,
        duration,
        cpu,
        mem,
    )


class PiperSynthesizer:
    """
    Wrapper around Piper TTS for personalized synthesis.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Piper synthesizer.

        Args:
            model_path: Path to Piper .onnx model file.
                       If None, tries to find default model.
        """
        self.voice = None
        self.model_path = model_path
        self.piper_ready = False
        self.sample_rate = 22050

        if PIPER_AVAILABLE:
            self._init_piper(model_path)
        else:
            logger.error("Piper not available!")

    def _init_piper(self, model_path: Optional[str] = None):
        """Initialize Piper voice model."""
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"Loading Piper model: {model_path}")
                self.voice = PiperVoice.load(model_path)
                self.piper_ready = True
                logger.info("Piper model loaded successfully")
                return

            # Try to find default model in common locations
            default_models = [
                Path.home() / ".local/share/piper/models/en_US-american_english-medium.onnx",
                Path.home() / ".local/share/piper/models/en_US-amy-medium.onnx",
                Path.home() / ".local/share/piper/models/en_US-lessac-medium.onnx",
                Path.home() / ".local/share/piper/models/en_US-libritts-high.onnx",
                Path("/usr/share/piper/models/en_US-american_english-medium.onnx"),
            ]

            for model in default_models:
                if model.exists():
                    logger.info(f"Found default model: {model}")
                    self.voice = PiperVoice.load(str(model))
                    self.model_path = str(model)
                    self.sample_rate = self.voice.config.sample_rate
                    self.piper_ready = True
                    logger.info("Piper model loaded successfully")
                    return

            logger.error("No Piper model found in default locations")
            logger.error(
                "   Download a model manually or set --model-path to a valid .onnx file"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Piper: {e}", exc_info=True)

    def is_ready(self) -> bool:
        """Check if Piper is ready for synthesis."""
        return self.piper_ready and self.voice is not None

    def synthesize(
        self,
        text: str,
        pitch_adjust: float = 1.0,
        speed_adjust: float = 1.0,
    ) -> Optional[bytes]:
        """
        Synthesize text to speech with personalization parameters.

        Returns raw WAV bytes.
        """
        if not self.is_ready():
            logger.error("Piper not ready. Cannot synthesize.")
            return None

        try:
            logger.info(
                f"Synthesizing: '{text[:50]}...' | pitch={pitch_adjust:.2f}, speed={speed_adjust:.2f}"
            )

            t0 = time.perf_counter()

            # Build synthesis config
            # Piper uses length_scale: >1.0 = slower, <1.0 = faster
            # We treat speed_adjust >1.0 as faster, so invert it.
            length_scale = 1.0 / max(0.1, speed_adjust)

            syn_config = SynthesisConfig(
                length_scale=length_scale,
                # You can tweak these or leave defaults:
                noise_scale=0.667,
                noise_w_scale=0.8,
                # volume=1.0,
                # speaker_id=0,
            )

            # Use an in-memory buffer via wave + BytesIO
            import io

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                # Piper will set channels, sample width, frame rate internally
                self.voice.synthesize_wav(
                    text,
                    wav_file,
                    syn_config=syn_config,
                )

            _log_perf("inference_synthesis", t0)

            audio_bytes = buffer.getvalue()
            logger.info(f"Synthesis complete ({len(audio_bytes)} bytes)")
            return audio_bytes

        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return None

    def save_audio(self, audio_bytes: bytes, output_path: str) -> bool:
        """
        Save audio bytes to file.

        Args:
            audio_bytes: Raw audio bytes (WAV format from Piper)
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            file_size_kb = len(audio_bytes) / 1024
            logger.info(f"Audio saved: {output_path} ({file_size_kb:.1f} KB)")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio: {e}", exc_info=True)
            return False