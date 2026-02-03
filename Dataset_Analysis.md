# TTS Dataset Analysis & Pipeline

## 1. Dataset Overview

| Dataset          | Duration | Speakers              | Audio Quality | Best For...                             |
| :--------------- | :------- | :-------------------- | :------------ | :-------------------------------------- |
| **LJ Speech**    | 24 hrs   | 1 (Female, US)        | 22kHz         | Beginners, Single Voice, Benchmarking   |
| **VCTK**         | 44 hrs   | 110 (English accents) | 48kHz         | Accents, Voice Conversion Research      |
| **LibriTTS**     | 585 hrs  | 2,400+ (Various)      | 24kHz         | Voice Cloning, Generalization, "Big AI" |
| **Hi-Fi TTS**    | 292 hrs  | 10 (Studio Pros)      | 44.1kHz       | Commercial Quality, High-Fidelity       |
| **HUI (German)** | 326 hrs  | 5 Major + 117 Minor   | 44.1kHz       | German Language SOTA                    |

## 2. Voice Quality Mapping

_How dataset properties mechanically force output characteristics._

| Attribute        | Input Property                           | Mechanism & Result                                                                                                                                                                                                                             |
| :--------------- | :--------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Clarity**      | **SNR / Noise Floor**                    | **Mechanism:** Model generates silence too.<br>**Result:** High noise input = "Static/Hiss" in output.                                                                                                                                         |
|                  | **Sample Rate**                          | **Mechanism:** The sample rate defines the highest pitch frequency the model can reproduce (The Nyquist Limit).<br>**Result:** 44.1kHz+ (22kHz limit): "Hi-Fi." Captures the breathiness and subtle mouth clicks that make audio sound "real." |
|                  | **Articulation**                         | **Mechanism:** Alignment/Attention.<br>**Result:** Mumbling input = Glitching/skipping words.                                                                                                                                                  |
| **Naturalness**  | **The "Human-ness" and flow. / Prosody** | **Mechanism:** Pitch variance stats.<br>**Result:** "Audiobook" style = Robotic/hypnotic rhythm.                                                                                                                                               |
|                  | **Breaths**                              | **Mechanism:** Silence trimming.<br>**Result:** No breaths in data = Rushed, breathless output.                                                                                                                                                |
| **Accent**       | **Phoneme Map**                          | **Mechanism:** Vowel shape learning.<br>**Result:** Mixed accents = "Ghost Accent" (unstable vowels).                                                                                                                                          |
| **Pitch/Timbre** | **F0 & Envelope**                        | **Mechanism:** Identity of the speaker / Physical vocal tract shape.<br>**Result:** Cannot change gender effectively without changing timbre.                                                                                                  |
| **Emotion**      | **Data Range**                           | **Mechanism:** Latent space boundaries.<br>**Result:** Neutral input = AI cannot scream or cry.                                                                                                                                                |

## 3. Dataset Selection Guidelines

_Choose based on your specific target use case._

| Dataset              | Gender & Age                | Accent                              | Emotion Range                         | Speaking Style                         | Best For (Target Profile)                                                                                      |
| :------------------- | :-------------------------- | :---------------------------------- | :------------------------------------ | :------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **LJ Speech**        | Female Adult (Young)        | US (North American Neutral)         | Low (Consistent, Neutral)             | Read / Formal (Non-fiction Audiobook)  | **AI Assistants**<br>Ideal for standard, pleasant, and clear "Siri-like" voices.                               |
| **LibriTTS** (Clean) | Mixed (M & F, Various Ages) | Mostly US (Some UK/Other)           | Medium (Storytelling dynamic)         | Read / Narrative (Fiction Audiobooks)  | **Narrators**<br>Best for reading long articles or books where natural flow matters more than perfect clarity. |
| **VCTK**             | Mixed (109 speakers)        | Global (Scottish, Indian, US, etc.) | Low (Newspaper reading)               | Read / Neutral (Short sentences)       | **Global/Accent Work**<br>Use when a specific regional accent or voice conversion capability is required.      |
| **Expresso** (New)   | Mixed (2 Male, 2 Female)    | US                                  | High / Stylized (Whisper, Projecting) | Conversational (Filled pauses, laughs) | **Chatbots**<br>The most human-sounding data available for casual, conversational interactions.                |
| **Hi-Fi TTS**        | Mixed (6 Male, 4 Female)    | US / EU                             | Medium (Professional)                 | Read / Studio (High Fidelity)          | **Commercial / Ads**<br>Use when audio quality and crispness are the highest priorities.                       |

## 4. Data Preparation Pipeline

### Phase 1: Preprocessing

- [ ] **Convert:** All files to WAV (16-bit PCM).
- [ ] **Resample:** Target rate (22050 Hz standard, 44100 Hz Hi-Fi).
- [ ] **Channels:** Merge to Mono.
- [ ] **Segment:** Slice to 1–10 seconds (split at pauses).

### Phase 2: Quality Filter

- [ ] **SNR:** >60dB (No background hiss/fans).
- [ ] **Length:** Discard <0.5s (too short) and >10s (too long).
- [ ] **Manual Check:** Listen for clicks, echoes, or breathing artifacts.

### Phase 3: Normalization

- [ ] **Loudness:** Normalize to -23 LUFS or -1.0 dB Peak.
- [ ] **Silence:** Trim start/end silence to exactly 0.1s.
- [ ] **Text:** Expand all non-words (`$5` -> "five dollars", `1990` -> "nineteen ninety").

### Phase 4: Alignment

- [ ] **MFA:** Generate TextGrids to map words to timestamps.
- [ ] **Phonemize:** Convert text to IPA/symbols (e.g., eSpeak-ng).
- [ ] **Validate:** Discard clips where text duration ≠ audio duration.

## 5. Visualizations

### 1. The TTS Dataset Pipeline Diagram

```mermaid
flowchart TD;
    %% Defines Colors
    classDef default fill:#fff,stroke:#333,stroke-width:1px;
    classDef final fill:#dfd,stroke:#0f0,stroke-width:2px;

    %% Audio Section
    subgraph Audio_Processing
        A1["Raw Audio Files"] --> B1["Convert to WAV/Mono"];
        B1 --> B2["Resample (22050Hz)"];
        B2 --> B3["Silence Trimming"];
        B3 --> B4["Loudness Normalization"];
    end

    %% Text Section
    subgraph Text_Processing
        C1["Raw Text Transcripts"] --> C2["Text Cleaning"];
        C2 --> C3["Normalize Text ($10 to ten dollars)"];
    end

    %% Logic Connections
    B4 --> D{"Alignment Check"};
    C3 --> D;

    %% Outcomes
    D -- "Bad Quality" --> E["Discard File"];
    D -- "Good Match" --> F["Final Formatting (JSON)"];

    F --> G[("READY TTS DATASET")]:::final;
```

### 2. Feature Extraction Flow (The Math Conversion)

```mermaid
flowchart TD
    subgraph TEXT_PATH [Text Processing]
        direction TB
        T1["Normalized Text"] --> T2["Phonemization<br/>(e.g. eSpeak-ng)"]
        T2 -- "h@l'oʊ" --> T3["ID Mapping / Tokenization"]
        T3 -- "[12, 45, 33, 9]" --> T4["Phoneme Embedding Sequence"]
    end

    subgraph AUDIO_PATH [Audio Processing]
        direction TB
        A1["Processed WAV"] --> A2["STFT / Short-Time<br/>Fourier Transform"]
        A2 --> A3["Mel-Filterbank"]
        A3 -- "Visual Heatmap" --> A4["Mel-Spectrogram"]
    end

    %% Connections to the Model
    T4 --> M["TTS MODEL TRAINER"]
    A4 --> M

    %% Styling
    style T4 fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000
    style A4 fill:#ffcccb,stroke:#d9534f,stroke-width:2px,color:#000
    style M fill:#cce5ff,stroke:#007bff,stroke-width:4px,color:#000
```

### 3. Voice Characteristic Mapping Diagram (Mindmap)

```mermaid
mindmap
  root((TARGET VOICE PROFILE))
    CLARITY
      High SNR (Studio Silence >60dB)
      High Sample Rate (44.1kHz for crispness)
      Precise Articulation (No mumbling)
    NATURALNESS
      Retained natural breaths
      Varied sentence duration
      Complex punctuation usage
    ACCENT
      Single, region-specific speaker
      Consistent vowel pronunciation
      Phonetically balanced script
    EMOTION RANGE
      Acted/Perfomed Data (Not just read)
      High Dynamic Range (Loud vs Soft contrast)
      Explicit Emotion Tags in metadata
    SPEAKING STYLE
      Conversational: Includes fillers , laughter, varying speed
      Narrative: Steady rhythm, perfect grammar, formal tone

```

# End
