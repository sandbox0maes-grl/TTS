import logging
import struct
import io
import time
import psutil
import wave
import os
from piper_synthesizer import PiperSynthesizer

logger = logging.getLogger(__name__)
process = psutil.Process()

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

class PersonalizationEngine(...):
    ...

    def synthesize_with_profile(
        self,
        profile_path: str,
        text: str,
        output_path: str,
        model_path: str | None = None,
    ) -> str:
        """
        Load a saved profile JSON, compute parameters, and synthesize audio.

        Returns:
            Path to the generated audio file.
        """
        # Load profile
        profile = self.load_profile(profile_path)
        user_id = profile.user_id

        # Get Piper parameters
        params = self.get_synthesis_parameters(user_id)

        # Initialize Piper
        synth = PiperSynthesizer(model_path=model_path)

        t0 = time.perf_counter()
        audio = synth.synthesize(
            text=text,
            pitch_adjust=params["pitch_adjust"],
            speed_adjust=params["speaking_rate_adjust"],
        )
        _log_perf("inference_synthesis", t0)

        synth.save_audio(audio, output_path)
        logger.info(
            "Generated personalized audio | user=%s | text_len=%d | file=%s",
            user_id,
            len(text),
            output_path,
        )
        return output_path
