# Personalized Speech Synthesis with Piper TTS

## Project Overview

This project extends **Piper TTS** (a fast, offline text-to-speech system) with **personalization capabilities**.

### What Does It Do?

**Simple Explanation:**

- Piper TTS can read text aloud in different voices
- Our system learns how **YOU** speak from your voice recordings
- Then it makes Piper sound like YOU when reading text

**Real-World Example:**

```
INPUT: "Hello, my name is Mahesh"
Piper (default): Reads with generic voice
Our System: Reads with Mahesh's unique voice patterns
            - speaking speed
            - pitch/accent
            - natural pauses
            - emotion/expression
```

---

## What's Included

```
piper-personalization/
├──
├── audio_preprocessor.py      # Cleans audio recordings
├── feature_extractor.py       # Learns speaking patterns
├── personalization_engine.py  # Main system that ties it together
├── user_audio.mp3             # Example audio files
├──
│   └── voice_profiles/        # Saved user voice profiles
|     └── user1_profile.json
├── personalization.log        # System logs (shows what happened)
├── README.md                  # This file
├── ARCHITECTURE.md            # Technical details
├── DATASET_ANALYSIS.md        # How datasets affect voice
└── requirements.txt           # Python dependencies
```

---

## Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Prepare Your Audio

Record yourself speaking for at least 30 seconds

**Requirements:**

- Format: MP3
- Quality: Clear audio, minimal background noise
- Content: Natural speech, varied sentences

Save as: `user_audio.mp3`

### Step 3: Create profile

Run the command to generate the user profile json file
py cli.py create-profile --user-id ip_user_id --user-name "ip_userName" --audio ip_user_audio.mp3

### Step 4: Synthesis of Audio

Run the command to generate the audio
python cli.py synthesize --user-id ip_user_id --text "Input Text" --output output.wav --model en_US-danny-low.onnx

User can use his own model generated with Piper Studio

## What Gets Extracted From Your Voice

The system learns and measures:

### 1. **Speaking Rate** (Words Per Minute)

```
Your recording: 140 WPM
Result: Piper reads at YOUR speed, not too fast/slow
```

### 2. **Pitch/Voice Height**

```
Your voice: 120 Hz (lower/deeper)
Result: Piper uses lower pitch to match you
```

### 3. **Pausing Patterns**

```
You pause: 0.5 seconds between sentences
Result: Piper pauses naturally like you do
```

### 4. **Energy Levels**

```
Your loudness: 65 dB average
Result: Piper synthesizes at your loudness level
```

### 5. **Pitch Contours**

```
Your intonation: Voice goes up at end of questions
Result: Piper copies your intonation patterns
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              PERSONALIZATION SYSTEM                 │
└─────────────────────────────────────────────────────┘
        ↑                          ↓
        │                          │
┌───────┴──────────┐    ┌─────────┴──────────┐
│   USER AUDIO     │    │  PERSONALIZATION   │
│   RECORDINGS     │    │     ENGINE         │
└───────┬──────────┘    └─────────┬──────────┘
        ↓                         ↓
    ┌──────────────────────────────────────┐
    │   AUDIO PREPROCESSOR                 │
    │  (Clean & prepare audio)             │
    └──────────┬───────────────────────────┘
               ↓
    ┌──────────────────────────────────────┐
    │   FEATURE EXTRACTOR                  │
    │  (Learn patterns: pitch, speed, etc) │
    └──────────┬───────────────────────────┘
               ↓
    ┌──────────────────────────────────────┐
    │   VOICE PROFILE                      │
    │  (Store all learned patterns)        │
    │  Format: JSON or YAML                │
    └──────────┬───────────────────────────┘
               ↓
    ┌──────────────────────────────────────┐
    │   TTS ADAPTER                        │
    │  (Convert to Piper parameters)       │
    └──────────┬───────────────────────────┘
               ↓
    ┌──────────────────────────────────────┐
    │   PIPER TTS                          │
    │  (Synthesize personalized speech)    │
    │  ↓                                   │
    │  OUTPUT: Personalized audio          │
    └──────────────────────────────────────┘
```

---

## Data Files & Formats

### Voice Profile (JSON)

```json
{
  "user_id": "user_001",
  "user_name": "Test User",
  "created_date": "2025-01-15T10:30:00",
  "speaking_characteristics": {
    "speaking_rate_wpm": 145.3,
    "average_pause_duration": 0.52,
    "mean_pitch": 115.4,
    "pitch_range": [85.2, 180.5]
  },
  "emotion_profile": {}
}
```

---

## 🔍 Understanding the Code

### Main Components

#### 1. **AudioPreprocessor** (`audio_preprocessor.py`)

**Purpose:** Clean up messy audio recordings

**Key Functions:**

- `load_audio()` - Load audio file
- `remove_silence()` - Remove quiet parts
- `normalize_audio()` - Make volume consistent
- `reduce_noise_simple()` - Remove background noise
- `preprocess()` - Run all steps

**Time Complexity:** O(n) where n = audio length in samples

#### 2. **FeatureExtractor** (`feature_extractor.py`)

**Purpose:** Analyze speaking patterns from audio

**Key Functions:**

- `extract_energy()` - Measure loudness
- `extract_pitch()` - Find voice pitch
- `extract_pauses()` - Detect silence/pauses
- `calculate_speaking_rate()` - Estimate WPM
- `extract_all_features()` - Run all extractions

**Time Complexity:** O(n log n) due to FFT for pitch detection

#### 3. **PersonalizationEngine** (`personalization_engine.py`)

**Purpose:** Orchestrate everything and create voice profiles

**Key Functions:**

- `create_voice_profile()` - Learn from audio
- `get_synthesis_parameters()` - Get Piper parameters
- `save_profile()` - Save to disk

---

## Logging System

The system logs everything for debugging and monitoring.

### Log File Location

```
personalization.log
```

### Example Log Output

```
2025-01-15 10:30:45 - audio_preprocessor - INFO - Loaded audio: user_001.wav
2025-01-15 10:30:45 - audio_preprocessor - INFO -   Duration: 300.45 seconds
2025-01-15 10:30:45 - audio_preprocessor - INFO -   Sample rate: 22050 Hz
2025-01-15 10:30:46 - feature_extractor - INFO - Energy extraction: 2341 frames
2025-01-15 10:30:47 - feature_extractor - INFO - Pitch extraction: 1950 voiced frames
2025-01-15 10:30:47 - feature_extractor - INFO -   Mean pitch: 115.42 Hz
2025-01-15 10:30:48 - personalization_engine - INFO - Voice profile created
```

### What Gets Logged

- ✓ Audio loading info (duration, sample rate)
- ✓ Processing steps (silence removal, normalization)
- ✓ Extracted features (pitch, energy, pauses)
- ✓ Performance metrics (processing time)
- ✓ Errors and warnings
- ✓ File operations (save/load)

## ⚙️ Configuration

### Audio Settings

```python
# In personalization_engine.py
engine = PersonalizationEngine(sr=22050)  # Sample rate

# sr options:
# 16000 - Lower quality, faster processing
# 22050 - Recommended (good balance)
# 44100 - Higher quality, more processing
# 48000 - Professional audio
```

### Feature Extraction Thresholds

```python
# In feature_extractor.py
threshold_percentile=25  # Quiet parts for pause detection

# Lower = more sensitive to pauses
# Higher = less sensitive to pauses
```

## 🐛 Troubleshooting

### Issue: "Failed to load audio file"

**Solution:** Check file format (WAV, MP3, FLAC) and path

### Issue: "No pitch detected"

**Solution:** Audio may be too quiet or noisy. Try re-recording

### Issue: "Processing is very slow"

**Solution:** Reduce sample rate (sr=16000) or audio length

### Issue: "Profiles not saving"

**Solution:** Check directory permissions, create ./profiles folder

---

## 📚 References

- **Piper TTS:** https://github.com/rhasspy/piper
- **Librosa:** https://librosa.org/
- **Speech Processing:** https://www.dsprelated.com/

**END**
