import numpy as np
import librosa
import logging
from dataclasses import dataclass, asdict
import json
import soundfile as sf
from pathlib import Path
import logging
import psutil
import os
logger = logging.getLogger(__name__)

@dataclass
class SpeakingPatterns:    
    # A container for all speaking patterns we extract from audio
    
    speaking_rate_wpm: float  # Words per minute
    average_pause_duration: float  # How long they pause
    energy_levels: list  # Loudness over time
    pitch_contour: list  # Pitch ups and downs
    mean_pitch: float  # Average pitch (Hz)
    pitch_range: tuple  # (lowest, highest) pitch
    

class FeatureExtractor:

    def __init__(self, sr=22050, **kwargs):
        self.sr = sr  # Sample rate
        logger.info(f"FeatureExtractor initialized with sample rate: {sr}")
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
               
    
    def extract_energy(self, audio):
        
        # Extract how LOUD the audio is at different times        
        # Calculate energy using framing
        frame_length = 2048  # Size of each chunk
        hop_length = 512  # How much to move forward each time
        
        # S = magnitude spectrogram (shows frequency content)
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr,
            n_fft=frame_length,
            hop_length=hop_length
        )
        
        # Convert to decibels 
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Average energy across frequencies for each time frame
        energy = np.mean(S_db, axis=0)
        
        logger.info(f"Energy extraction: {len(energy)} frames")
        logger.info(f"  Energy range: {energy.min():.2f} to {energy.max():.2f} dB")
        
        return energy
    
    def extract_pitch(self, audio):
        
        # pyin for pitch extraction
        # f0 = fundamental frequency
        # voiced_flag = voice/silence
        # voiced_probs = confidence in voice
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # Lowest possible pitch
            fmax=librosa.note_to_hz('C7')   # Highest possible pitch
        )
        
        # Remove unvoiced (silence/noise) regions
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            logger.warning("No pitch detected - possibly silent audio")
            return None
        
        # Calculate statistics
        mean_pitch = np.nanmean(f0_voiced)
        std_pitch = np.nanstd(f0_voiced)
        min_pitch = np.nanmin(f0_voiced)
        max_pitch = np.nanmax(f0_voiced)
        
        logger.info(f"Pitch extraction: {len(f0_voiced)} voiced frames")
        logger.info(f"  Mean pitch: {mean_pitch:.2f} Hz")
        logger.info(f"  Pitch range: {min_pitch:.2f} - {max_pitch:.2f} Hz")
        logger.info(f"  Pitch std dev: {std_pitch:.2f} Hz")
        
        return f0_voiced, (min_pitch, max_pitch, mean_pitch, std_pitch)
    
    def extract_pauses(self, audio, energy, threshold_percentile=25):
        
        # Find where the person PAUSES (stops talking)        
        # Calculate threshold for pause detection
        # 25th percentile = if sorted by loudness, the quietest 25%
        threshold = np.percentile(energy, threshold_percentile)
        
        # Find frames below threshold 
        pause_frames = energy < threshold
        
        # Group consecutive pause frames together
        hop_length = 512
        frame_times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=self.sr,
            hop_length=hop_length
        )
        
        # Find pause start and end points
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        for i, is_paused in enumerate(pause_frames):
            if is_paused and not in_pause:
                # Pause starts
                pause_start = frame_times[i]
                in_pause = True
            elif not is_paused and in_pause:
                # Pause ends
                pause_end = frame_times[i]
                duration = pause_end - pause_start
                pause_durations.append(duration)
                in_pause = False
        
        if len(pause_durations) > 0:
            avg_pause = np.mean(pause_durations)
            logger.info(f"Pause detection: {len(pause_durations)} pauses found")
            logger.info(f"  Average pause duration: {avg_pause:.3f} seconds")
        else:
            logger.info("No significant pauses detected")
        
        return pause_durations
    
    def calculate_speaking_rate(self, audio, energy, pause_durations):
        '''
        1. Calculate total audio duration
        2. Subtract all pauses (non-speaking time)
        3. Estimate words based on remaining time

        Typical rates:
        - 100-150 WPM = normal
        - <100 WPM = slow
        - >150 WPM = fast
        '''
        # Total audio duration in seconds
        total_duration = len(audio) / self.sr
        
        # Total pause time
        total_pause_time = sum(pause_durations) if pause_durations else 0
        
        # Speaking time = total - pauses
        speaking_time = total_duration - total_pause_time
        
        # Rough estimate: assume 4-5 phonemes per word on average
        # This is a simplified model - real systems are more complex
        estimated_words = speaking_time * 2.5  # ~2.5 words per second
        estimated_wpm = (estimated_words / speaking_time * 60) if speaking_time > 0 else 0
        
        logger.info(f"Speaking rate calculation:")
        logger.info(f"  Total duration: {total_duration:.2f} seconds")
        logger.info(f"  Pause time: {total_pause_time:.2f} seconds")
        logger.info(f"  Speaking time: {speaking_time:.2f} seconds")
        logger.info(f"  Estimated WPM: {estimated_wpm:.1f}")
        
        return estimated_wpm
    
    def extract_all_features(self, audio):

        # It runs all extraction steps and returns a nice organized package with everything.
        
        logger.info("Starting complete feature extraction")
        
        # Step 1: Extract energy (loudness)
        energy = self.extract_energy(audio)
        logger.info("Step 1: Energy extracted")
        self.log_resources("Step 1: Energy extracted")
        
        # Step 2: Extract pitch
        pitch_data = self.extract_pitch(audio)
        if pitch_data is None:
            logger.error("Step 2: Failed to extract pitch")
            return None
        pitch_values, pitch_stats = pitch_data
        logger.info("Step 2: Pitch extracted")
        self.log_resources("Step 2: Pitch extracted")
        
        # Step 3: Extract pauses
        pauses = self.extract_pauses(audio, energy)
        logger.info("Step 3: Extract pauses")
        self.log_resources("Step 3: Extract pauses")
        
        # Step 4: Calculate speaking rate
        wpm = self.calculate_speaking_rate(audio, energy, pauses)
        logger.info("Step 4: Speaking rate calculated")
        self.log_resources("Step 4: Speaking rate calculated")
        
        # Compile everything into a pattern object
        patterns = SpeakingPatterns(
            speaking_rate_wpm=wpm,
            average_pause_duration=np.mean(pauses) if pauses else 0,
            energy_levels=energy.tolist(),
            pitch_contour=pitch_values.tolist(),
            mean_pitch=pitch_stats[2],
            pitch_range=(pitch_stats[0], pitch_stats[1])
        )
        
        logger.info("All features extracted successfully!")
        self.log_resources("All features extracted successfully!")
        
        return patterns


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create extractor
    extractor = FeatureExtractor(sr=22050)
    
    # first preprocess audio with AudioPreprocessor
    try:
        # Load a sample audio file
        audio, sr = librosa.load('user_audio_cleaned.mp3', sr=22050)
        
        # Extract all features
        patterns = extractor.extract_all_features(audio)
        
        if patterns:
            print("\n" + "="*50)
            print("EXTRACTED SPEAKING PATTERNS:")
            print("="*50)
            print(f"Speaking Rate: {patterns.speaking_rate_wpm:.1f} WPM")
            print(f"Average Pause: {patterns.average_pause_duration:.3f} seconds")
            print(f"Mean Pitch: {patterns.mean_pitch:.2f} Hz")
            print(f"Pitch Range: {patterns.pitch_range[0]:.2f} - {patterns.pitch_range[1]:.2f} Hz")
            print("="*50)
            
            # Save to JSON file
            with open('speaking_patterns.json', 'w') as f:
                json.dump(asdict(patterns), f, indent=2)
            print("Patterns saved to speaking_patterns.json")
    
    except FileNotFoundError:
        print("Note: Please provide a sample audio file named 'user_audio_cleaned.mp3'")