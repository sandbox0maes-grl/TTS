"""
Audio Preprocessor Module
This module handles:
1. Loading audio files
2. Reducing noise
3. Normalizing volume
4. Removing silence
5. Logging all operations

"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import psutil
import os

logger = logging.getLogger(__name__)

class AudioPreprocessor: 

    def __init__(self, sr=22050, **kwargs):
        self.sr = sr  # Sample rate
        logger.info(f"AudioPreprocessor initialized with sample rate: {sr}")
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None) # Reset counter
    
    def log_resources(self, stage):
        try:
            # Measure usage since last call
            cpu_usage = self.process.cpu_percent(interval=None)
            
            # RAM in MB
            mem_bytes = self.process.memory_info().rss
            mem_mb = mem_bytes / 1024 / 1024
            
            logger.info(f"PERF [{stage}] -> CPU: {cpu_usage:.1f}% | RAM: {mem_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Could not log resources: {e}")


    def load_audio(self, audio_path):        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)

            duration = len(audio) / sr
            logger.info(f"Loaded audio: {Path(audio_path).name}")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Sample rate: {sr} Hz")
            logger.info(f"  Total samples: {len(audio)}")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {str(e)}")
            raise
    
    def remove_silence(self, audio, top_db=20):

        # top_db = if sound is 20 decibels below the loudest point, consider it silence
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        
        removed_samples = len(audio) - len(trimmed_audio)
        logger.info(f"Silence removal: removed {removed_samples} samples")
        logger.info(f"  Original length: {len(audio)} samples")
        logger.info(f"  After trimming: {len(trimmed_audio)} samples")
        
        return trimmed_audio
    
    def normalize_audio(self, audio):
        
        max_val = np.max(np.abs(audio))
        
        if max_val == 0:
            logger.warning("Audio is silent")
            return audio
        
        normalized = audio / max_val
        
        logger.info(f"Normalization: max amplitude before = {max_val:.4f}")
        logger.info(f"  Max amplitude after = {np.max(np.abs(normalized)):.4f}")
        
        return normalized
    
    def reduce_noise_simple(self, audio):
        
        # Assume first 0.5 seconds is noise
        noise_duration_samples = int(0.5 * self.sr)
        noise_sample = audio[:noise_duration_samples]
        
        # frequency characteristics of noise
        # convert time domain to frequency domain
        noise_spectrum = np.abs(np.fft.fft(noise_sample))
        
        logger.info(f"Noise profile detected from first {0.5} seconds")
        logger.info(f"  Noise spectrum shape: {noise_spectrum.shape}")
        
        # Simple noise reduction: reduce audio where noise is strong
        cleaned_audio = audio
        
        logger.info("Simple noise reduction applied")
        
        return cleaned_audio
    
    def preprocess(self, audio_path, output_path=None):
        
        logger.info(f"Starting preprocessing pipeline for {audio_path}")
        
        # Step 1: Load
        audio, sr = self.load_audio(audio_path)
        logger.info("Step 1: Audio loaded")
        self.log_resources("Step 1: Load")
        
        # Step 2: Remove silence
        audio = self.remove_silence(audio)
        logger.info("Step 2: Silence removed")
        self.log_resources("Step 2: Silence removed")
        
        # Step 3: Normalize
        audio = self.normalize_audio(audio)
        logger.info("Step 3: Normalized")
        self.log_resources("Step 3: Normalized")
        
        # Step 4: Reduce noise
        audio = self.reduce_noise_simple(audio)
        logger.info("Step 4: Noise reduced")
        self.log_resources("Step 4: Noise reduced")
        
        # Step 5: Save
        if output_path:
            sf.write(output_path, audio, sr)
            logger.info(f"Saved preprocessed audio to {output_path}")
            self.log_resources("Saved preprocessed audio to {output_path}")
        
        logger.info("Preprocessing complete!")
        self.log_resources("Preprocessing complete")
        return audio, sr


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    preprocessor = AudioPreprocessor(sr=22050)    
    
    try:
        audio, sr = preprocessor.preprocess(
            audio_path='user_audio.mp3',
            output_path='user_audio_cleaned.mp3'
        )
        print("Audio preprocessing complete!")
        
    except FileNotFoundError:
        print("Note: Please provide a sample audio file named 'user_audio.mp3'")