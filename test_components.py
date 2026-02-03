"""
Unit Tests for Personalization Engine
======================================

HUMAN LANGUAGE EXPLANATION:
These tests verify that each component works correctly.
Think of them as quality checks - ensuring everything functions as expected.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Import the components to test
from audio_preprocessor import AudioPreprocessor
from feature_extractor import FeatureExtractor, SpeakingPatterns
from personalization_engine import PersonalizationEngine, VoiceProfile


# ============================================================================
# TEST AUDIO PREPROCESSOR
# ============================================================================

class TestAudioPreprocessor:
    """Test the audio cleaning functionality"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor for testing"""
        return AudioPreprocessor(sr=22050)
    
    @pytest.fixture
    def synthetic_audio(self):
        """Create synthetic audio for testing (no file needed)"""
        sr = 22050
        duration = 2  # 2 seconds
        
        # Generate a simple sine wave at 440 Hz (A note)
        t = np.linspace(0, duration, sr * duration)
        freq = 440
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        return audio, sr
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test that preprocessor initializes correctly"""
        assert preprocessor.sr == 22050
        print("✓ Preprocessor initialized with correct sample rate")
    
    def test_normalize_audio(self, preprocessor, synthetic_audio):
        """Test audio normalization"""
        audio, sr = synthetic_audio
        
        # Normalize
        normalized = preprocessor.normalize_audio(audio)
        
        # Check that max value is 1.0
        assert np.max(np.abs(normalized)) <= 1.0, "Max should be <= 1.0"
        assert np.max(np.abs(normalized)) > 0.99, "Max should be close to 1.0"
        print("✓ Audio normalized correctly")
    
    def test_remove_silence(self, preprocessor, synthetic_audio):
        """Test silence removal"""
        audio, sr = synthetic_audio
        
        # Add silence (zeros) at beginning and end
        silence = np.zeros(sr)  # 1 second of silence
        audio_with_silence = np.concatenate([silence, audio, silence])
        
        # Remove silence
        trimmed = preprocessor.remove_silence(audio_with_silence, top_db=20)
        
        # Check that audio got shorter
        assert len(trimmed) < len(audio_with_silence), "Should remove silence"
        assert len(trimmed) > 0, "Should have audio remaining"
        print("✓ Silence removal working correctly")
    
    def test_preprocess_pipeline(self, preprocessor, synthetic_audio):
        """Test complete preprocessing pipeline"""
        audio, sr = synthetic_audio
        
        # We can't test with actual files in unit tests, but we can
        # test the individual steps
        audio = preprocessor.normalize_audio(audio)
        audio = preprocessor.remove_silence(audio)
        
        # Verify audio properties
        assert len(audio) > 0, "Should have audio after preprocessing"
        assert np.max(np.abs(audio)) <= 1.0, "Should be normalized"
        print("✓ Preprocessing pipeline works")


# ============================================================================
# TEST FEATURE EXTRACTOR
# ============================================================================

class TestFeatureExtractor:
    """Test the feature extraction functionality"""
    
    @pytest.fixture
    def extractor(self):
        """Create an extractor for testing"""
        return FeatureExtractor(sr=22050)
    
    @pytest.fixture
    def synthetic_speech(self):
        """Create synthetic speech-like audio"""
        sr = 22050
        duration = 3
        t = np.linspace(0, duration, sr * duration)
        
        # Create varying pitch (simulating speech)
        # Start at 100 Hz, go to 120 Hz, back to 100 Hz
        pitch = 100 + 20 * np.sin(2 * np.pi * 0.5 * t)
        
        # Create audio with varying pitch
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        audio = 0.3 * np.sin(phase)
        
        return audio, sr
    
    def test_extractor_initialization(self, extractor):
        """Test that extractor initializes correctly"""
        assert extractor.sr == 22050
        print("✓ Extractor initialized correctly")
    
    def test_energy_extraction(self, extractor, synthetic_speech):
        """Test energy extraction"""
        audio, sr = synthetic_speech
        
        energy = extractor.extract_energy(audio)
        
        # Check that energy was extracted
        assert isinstance(energy, np.ndarray), "Should return numpy array"
        assert len(energy) > 0, "Should have energy values"
        assert np.all(energy <= 0), "Energy in dB should be <= 0"
        print("✓ Energy extraction working")
    
    def test_pitch_extraction(self, extractor, synthetic_speech):
        """Test pitch extraction"""
        audio, sr = synthetic_speech
        
        result = extractor.extract_pitch(audio)
        
        if result is not None:
            pitch_values, pitch_stats = result
            
            # Check pitch values
            assert len(pitch_values) > 0, "Should have pitch values"
            assert np.all(pitch_values > 0), "Pitch should be positive (Hz)"
            
            # Check pitch statistics
            min_pitch, max_pitch, mean_pitch, std_pitch = pitch_stats
            assert min_pitch > 0, "Min pitch should be positive"
            assert max_pitch > min_pitch, "Max should be > min"
            assert mean_pitch > 0, "Mean should be positive"
            
            print(f"✓ Pitch extraction working (mean: {mean_pitch:.1f} Hz)")
        else:
            print("⚠ Pitch extraction returned None (expected for some audio)")
    
    def test_pause_extraction(self, extractor, synthetic_speech):
        """Test pause detection"""
        audio, sr = synthetic_speech
        energy = extractor.extract_energy(audio)
        
        pauses = extractor.extract_pauses(audio, energy)
        
        # Pauses should be a list
        assert isinstance(pauses, list), "Should return list"
        # For synthetic continuous audio, may not detect pauses
        if len(pauses) > 0:
            assert all(p > 0 for p in pauses), "Pause durations should be positive"
            print(f"✓ Pause detection working ({len(pauses)} pauses found)")
        else:
            print("✓ Pause detection working (no pauses in continuous audio)")
    
    def test_speaking_rate_calculation(self, extractor, synthetic_speech):
        """Test speaking rate calculation"""
        audio, sr = synthetic_speech
        energy = extractor.extract_energy(audio)
        pauses = extractor.extract_pauses(audio, energy)
        
        wpm = extractor.calculate_speaking_rate(audio, energy, pauses)
        
        # WPM should be positive
        assert wpm >= 0, "WPM should be non-negative"
        assert wpm < 500, "WPM should be reasonable"
        print(f"✓ Speaking rate calculated: {wpm:.1f} WPM")
    
    def test_extract_all_features(self, extractor, synthetic_speech):
        """Test complete feature extraction"""
        audio, sr = synthetic_speech
        
        patterns = extractor.extract_all_features(audio)
        
        if patterns is not None:
            # Check that all fields are present
            assert isinstance(patterns, SpeakingPatterns), "Should return SpeakingPatterns"
            assert patterns.speaking_rate_wpm >= 0, "WPM should be non-negative"
            assert patterns.mean_pitch > 0, "Mean pitch should be positive"
            assert len(patterns.energy_levels) > 0, "Should have energy values"
            assert len(patterns.pitch_contour) > 0, "Should have pitch values"
            
            print("✓ All features extracted successfully")
            print(f"  - Speaking rate: {patterns.speaking_rate_wpm:.1f} WPM")
            print(f"  - Mean pitch: {patterns.mean_pitch:.1f} Hz")
            print(f"  - Pitch range: {patterns.pitch_range[0]:.1f} - {patterns.pitch_range[1]:.1f} Hz")
        else:
            print("⚠ Feature extraction returned None")


# ============================================================================
# TEST VOICE PROFILE
# ============================================================================

class TestVoiceProfile:
    """Test voice profile creation and storage"""
    
    def test_voice_profile_creation(self):
        """Test creating a voice profile"""
        profile = VoiceProfile(user_id="test_001", user_name="Test User")
        
        assert profile.user_id == "test_001"
        assert profile.user_name == "Test User"
        assert profile.created_date is not None
        print("✓ Voice profile created successfully")
    
    def test_profile_to_dict(self):
        """Test converting profile to dictionary"""
        profile = VoiceProfile(user_id="test_001", user_name="Test User")
        profile.speaking_rate_wpm = 150.0
        profile.mean_pitch = 120.0
        
        profile_dict = profile.to_dict()
        
        assert profile_dict['user_id'] == "test_001"
        assert profile_dict['speaking_characteristics']['speaking_rate_wpm'] == 150.0
        print("✓ Profile converted to dictionary successfully")
    
    def test_profile_save_load_json(self):
        """Test saving and loading profile as JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save profile
            profile = VoiceProfile(user_id="test_001", user_name="Test User")
            profile.speaking_rate_wpm = 150.0
            profile.mean_pitch = 120.0
            profile.pitch_range = (85.0, 180.0)
            
            filepath = Path(tmpdir) / "profile.json"
            profile.save_json(filepath)
            
            # Verify file exists
            assert filepath.exists(), "File should be created"
            
            # Load and verify
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['user_id'] == "test_001"
            assert loaded_data['speaking_characteristics']['speaking_rate_wpm'] == 150.0
            print("✓ Profile saved and loaded successfully")


# ============================================================================
# TEST PERSONALIZATION ENGINE
# ============================================================================

class TestPersonalizationEngine:
    """Test the main personalization engine"""
    
    @pytest.fixture
    def engine(self):
        """Create an engine for testing"""
        return PersonalizationEngine(sr=22050)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine.sr == 22050
        assert len(engine.voice_profiles) == 0
        print("✓ Engine initialized successfully")
    
    def test_get_synthesis_parameters(self, engine):
        """Test synthesis parameter generation"""
        # Create a dummy profile
        profile = VoiceProfile(user_id="test", user_name="Test")
        profile.speaking_rate_wpm = 150.0
        profile.mean_pitch = 120.0
        profile.average_pause_duration = 0.5
        profile.energy_levels = [40, 45, 50]
        
        engine.voice_profiles["test"] = profile
        
        # Get parameters
        params = engine.get_synthesis_parameters("test")
        
        assert params is not None, "Should return parameters"
        assert 'pitch_adjust' in params, "Should have pitch adjustment"
        assert 'speaking_rate_adjust' in params, "Should have rate adjustment"
        assert 0.5 <= params['pitch_adjust'] <= 2.0, "Pitch should be in valid range"
        assert 0.5 <= params['speaking_rate_adjust'] <= 2.0, "Rate should be in valid range"
        
        print("✓ Synthesis parameters generated correctly")
        print(f"  - Pitch adjust: {params['pitch_adjust']:.2f}")
        print(f"  - Speed adjust: {params['speaking_rate_adjust']:.2f}")
    
    def test_normalize_pitch_for_synthesis(self, engine):
        """Test pitch normalization"""
        # Test various pitch values
        pitch_80 = engine._normalize_pitch_for_synthesis(80)  # Low (male)
        pitch_150 = engine._normalize_pitch_for_synthesis(150)  # Medium
        pitch_220 = engine._normalize_pitch_for_synthesis(220)  # High (female)
        
        # All should be in valid range
        assert 0.5 <= pitch_80 <= 2.0, "Normalized pitch should be in range"
        assert 0.5 <= pitch_150 <= 2.0, "Normalized pitch should be in range"
        assert 0.5 <= pitch_220 <= 2.0, "Normalized pitch should be in range"
        
        # Lower pitch should have lower adjustment
        assert pitch_80 < pitch_220, "Lower pitch should have lower adjustment"
        
        print("✓ Pitch normalization working correctly")
        print(f"  - 80 Hz → {pitch_80:.2f}")
        print(f"  - 150 Hz → {pitch_150:.2f}")
        print(f"  - 220 Hz → {pitch_220:.2f}")
    
    def test_normalize_speed_for_synthesis(self, engine):
        """Test speed normalization"""
        speed_100 = engine._normalize_speed_for_synthesis(100)  # Slow
        speed_150 = engine._normalize_speed_for_synthesis(150)  # Normal
        speed_200 = engine._normalize_speed_for_synthesis(200)  # Fast
        
        # All should be in valid range
        assert 0.5 <= speed_100 <= 2.0, "Speed should be in valid range"
        assert 0.5 <= speed_150 <= 2.0, "Speed should be in valid range"
        assert 0.5 <= speed_200 <= 2.0, "Speed should be in valid range"
        
        # Slower should have lower adjustment
        assert speed_100 < speed_200, "Slower speech should have lower adjustment"
        
        print("✓ Speed normalization working correctly")
        print(f"  - 100 WPM → {speed_100:.2f}")
        print(f"  - 150 WPM → {speed_150:.2f}")
        print(f"  - 200 WPM → {speed_200:.2f}")
    
    def test_profile_save_and_load(self, engine):
        """Test saving and loading profiles"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profile
            profile = VoiceProfile(user_id="test_001", user_name="Test User")
            profile.speaking_rate_wpm = 150.0
            profile.mean_pitch = 120.0
            engine.voice_profiles["test_001"] = profile
            
            # Save profile
            path = engine.save_profile("test_001", tmpdir, format='json')
            assert path is not None, "Should return path"
            
            # Clear profiles
            engine.voice_profiles.clear()
            assert len(engine.voice_profiles) == 0
            
            # Load profile
            loaded = engine.load_profile(path)
            assert loaded is not None, "Should load profile"
            assert loaded.user_id == "test_001"
            assert loaded.speaking_rate_wpm == 150.0
            
            print("✓ Profile save and load working correctly")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test components working together"""
    
    def test_preprocessing_to_features(self):
        """Test audio preprocessing followed by feature extraction"""
        # Create synthetic audio
        sr = 22050
        duration = 2
        t = np.linspace(0, duration, sr * duration)
        
        # Varying frequency audio
        pitch = 100 + 30 * np.sin(2 * np.pi * 0.3 * t)
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        audio = 0.3 * np.sin(phase)
        
        # Preprocess
        preprocessor = AudioPreprocessor()
        audio = preprocessor.normalize_audio(audio)
        audio = preprocessor.remove_silence(audio)
        
        # Extract features
        extractor = FeatureExtractor()
        patterns = extractor.extract_all_features(audio)
        
        # Verify results
        assert patterns is not None
        assert patterns.speaking_rate_wpm >= 0
        assert patterns.mean_pitch > 0
        
        print("✓ Integration test passed: preprocessing → features")
    
    def test_full_personalization_pipeline(self):
        """Test the complete personalization pipeline"""
        # Create synthetic audio
        sr = 22050
        duration = 2
        t = np.linspace(0, duration, sr * duration)
        
        pitch = 120 + 30 * np.sin(2 * np.pi * 0.3 * t)
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        audio = 0.3 * np.sin(phase)
        
        # Create engine
        engine = PersonalizationEngine()
        
        # Process would go: preprocessor → features → profile
        # (Skipping file I/O for unit test)
        preprocessor = AudioPreprocessor()
        extractor = FeatureExtractor()
        
        audio = preprocessor.normalize_audio(audio)
        audio = preprocessor.remove_silence(audio)
        patterns = extractor.extract_all_features(audio)
        
        # Create profile
        profile = VoiceProfile(user_id="test", user_name="Test")
        profile.speaking_rate_wpm = patterns.speaking_rate_wpm
        profile.mean_pitch = patterns.mean_pitch
        profile.average_pause_duration = patterns.average_pause_duration
        profile.energy_levels = patterns.energy_levels
        profile.pitch_contour = patterns.pitch_contour
        
        engine.voice_profiles["test"] = profile
        
        # Get synthesis parameters
        params = engine.get_synthesis_parameters("test")
        
        assert params is not None
        assert 0.5 <= params['pitch_adjust'] <= 2.0
        assert 0.5 <= params['speaking_rate_adjust'] <= 2.0
        
        print("✓ Full personalization pipeline works!")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PERSONALIZATION ENGINE - UNIT TESTS")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])

