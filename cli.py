"""
Command-line interface for Piper TTS Personalization Engine.
Usage:
    py cli.py create-profile --user-id mahesh1 --user-name "Mahesh" --audio user_audio.mp3
    py cli.py synthesize --user-id mahesh1 --text "Hello world test text from onnx local model" --output output.wav --model en_US-danny-low.onnx
    py cli.py synthesize --user-id mahesh1 --text_file input_TTS.txt --output output.wav --model en_US-danny-low.onnx
"""

import argparse
import sys
import logging
from pathlib import Path
from personalization_engine import PersonalizationEngine
from piper_synthesizer import PiperSynthesizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cli.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def cmd_create_profile(args):    
    #Create a voice profile from user audio.    
    print("\n" + "="*60)
    print("CREATE VOICE PROFILE FROM AUDIO")
    print("="*60)
    
    try:
        # Validate inputs
        if not Path(args.audio).exists():
            print(f"Error: Audio file not found: {args.audio}")
            return False
        
        # Initialize engine
        print(f"\nInitializing personalization engine...")
        engine = PersonalizationEngine(sr=22050)
        
        # Create profile from audio
        print(f" Creating profile from: {args.audio}")
        profile = engine.create_voice_profile(
            user_id=args.user_id,
            user_name=args.user_name,
            audio_path=args.audio,
        )
        
        if not profile:
            print("Failed to create profile")
            return False
        
        # Display learned characteristics
        print(f"\nProfile Analysis Results:")
        print(f"   User ID              : {profile.user_id}")
        print(f"   User Name            : {profile.user_name}")
        print(f"   Speaking Rate        : {profile.speaking_rate_wpm:.1f} WPM")
        print(f"   Average Pause        : {profile.average_pause_duration:.3f} seconds")
        print(f"   Mean Pitch           : {profile.mean_pitch:.1f} Hz")
        print(f"   Pitch Range          : {profile.pitch_range[0]:.1f} - {profile.pitch_range[1]:.1f} Hz")
        print(f"   Detected Emotion     : {profile.emotion_profile.get('dominant_emotion', 'N/A')}")
        print(f"   Emotion Confidence   : {profile.emotion_profile.get('confidence', 0):.2f}")
        
        # Generate Piper parameters
        print(f"\nGenerating Piper TTS parameters...")
        params = engine.get_synthesis_parameters(args.user_id)
        print(f"   Pitch Adjustment     : {params['pitch_adjust']:.2f}")
        print(f"   Speed Adjustment     : {params['speaking_rate_adjust']:.2f}")
        print(f"   Pause Duration       : {params['pause_duration']:.3f} s")
        print(f"   Energy Level         : {params['energy_level']:.2f}")
        
        # Save profile
        print(f"\nSaving profile...")
        profiles_dir = Path("./profiles")
        profiles_dir.mkdir(exist_ok=True)
        json_path = engine.save_profile(args.user_id, str(profiles_dir), format='json')
        print(f"  Profile saved: {json_path}")
        
        print("\n" + "="*60)
        print("PROFILE CREATED SUCCESSFULLY")
        print("="*60)
        print(f"\nNext step: Use this profile to synthesize speech:")
        print(f"  python cli.py synthesize --user-id {args.user_id} --text \"Your text here\" --output output.wav")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating profile: {e}", exc_info=True)
        print(f"Error: {e}")
        return False


def get_text_from_args(args):
    """Helper function to extract text from either --text or --text_file"""
    if hasattr(args, 'text_file') and args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file '{args.text_file}': {e}")
            return None
    return args.text


def cmd_synthesize(args):    
    # Synthesize text using a personalized voice profile.    
    print("\n" + "="*60)
    print("SYNTHESIZE SPEECH WITH PERSONALIZED PROFILE")
    print("="*60)

    try:
        # Resolve the text input (either raw string or from file)
        actual_text = get_text_from_args(args)
        if not actual_text:
            print("Error: No text provided or text file is empty.")
            return False

        # Check if profile exists
        profile_path = Path(f"./profiles/{args.user_id}_profile.json")
        if not profile_path.exists():
            print(f"Error: Profile not found: {profile_path}")
            print(f" Create a profile first using: python cli.py create-profile --user-id {args.user_id} ...")
            return False

        # Initialize engine
        print(f"\nInitializing engine...")
        engine = PersonalizationEngine(sr=22050)

        # Synthesize using profile
        print(f"\nSynthesizing with personalized profile...")
        output_file = engine.synthesize_with_profile(
            user_id=args.user_id,
            text=actual_text,
            output_path=args.output,
            model_path=args.model if hasattr(args, 'model') else None,
        )

        if not output_file:
            print("Synthesis failed")
            return False

        print(f"\nAudio generated successfully!")
        print(f"   File: {output_file}")
        print(f"   Size: {Path(output_file).stat().st_size / 1024:.1f} KB")

        print("\n" + "="*60)
        print("SYNTHESIS COMPLETE")
        print("="*60)
        return True

    except Exception as e:
        logger.error(f"Error synthesizing: {e}", exc_info=True)
        print(f"Error: {e}")
        return False


def cmd_compare(args):    
    # Generate side-by-side comparison of original vs personalized synthesis.   
    
    print("\n" + "="*60)
    print("COMPARE: ORIGINAL VS PERSONALIZED SYNTHESIS")
    print("="*60)
    
    try:
        # Resolve the text input (either raw string or from file)
        actual_text = get_text_from_args(args)
        if not actual_text:
            print("Error: No text provided or text file is empty.")
            return False

        # Check if profile exists
        profile_path = Path(f"./profiles/{args.user_id}_profile.json")
        if not profile_path.exists():
            print(f"Error: Profile not found: {profile_path}")
            return False
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print(f"\nLoading profile and initializing Piper...")
        engine = PersonalizationEngine(sr=22050)
        profile = engine.load_profile(str(profile_path))
        params = engine.get_synthesis_parameters(args.user_id)
        synthesizer = PiperSynthesizer()
        
        # Generate original (default parameters)
        print(f"\nGenerating ORIGINAL synthesis (default Piper)...")
        original_audio = synthesizer.synthesize(
            text=actual_text,
            pitch_adjust=1.0,  # default
            speed_adjust=1.0,  # default
        )
        
        if original_audio:
            original_path = output_dir / "original.wav"
            synthesizer.save_audio(original_audio, str(original_path))
            print(f"Original saved: {original_path}")
        
        # Generate personalized
        print(f"\nGenerating PERSONALIZED synthesis ({profile.user_name})...")
        personalized_audio = synthesizer.synthesize(
            text=actual_text,
            pitch_adjust=params['pitch_adjust'],
            speed_adjust=params['speaking_rate_adjust'],
        )
        
        if personalized_audio:
            personalized_path = output_dir / "personalized.wav"
            synthesizer.save_audio(personalized_audio, str(personalized_path))
            print(f"Personalized saved: {personalized_path}")
        
        # Generate comparison report
        print(f"\nGenerating comparison report...")
        
        # Ensure text in report doesn't get too long if passing a whole file
        display_text = actual_text if len(actual_text) < 100 else actual_text[:97] + "..."
        
        report = f"""
            COMPARISON REPORT: Original vs Personalized Synthesis
            =====================================================

            Text: "{display_text}"
            Profile: {profile.user_name}
            Date: {profile.last_updated}

            ORIGINAL (Default Piper):
            - Pitch Adjustment: 1.0x (normal)
            - Speed Adjustment: 1.0x (normal)
            - File: original.mp3

            PERSONALIZED ({profile.user_name}):
            - Pitch Adjustment: {params['pitch_adjust']:.2f}x
            - Speed Adjustment: {params['speaking_rate_adjust']:.2f}x
            - Mean Pitch: {profile.mean_pitch:.1f} Hz
            - Speaking Rate: {profile.speaking_rate_wpm:.1f} WPM
            - Detected Emotion: {profile.emotion_profile.get('dominant_emotion', 'N/A')}
            - File: personalized.mp3

            DIFFERENCE:
            - Pitch changed by: {(params['pitch_adjust'] - 1.0) * 100:+.1f}%
            - Speed changed by: {(params['speaking_rate_adjust'] - 1.0) * 100:+.1f}%

            CHARACTERISTICS:
            {profile.user_name} speaks with:
            - Pitch: {profile.pitch_range[0]:.1f}-{profile.pitch_range[1]:.1f} Hz (range: {profile.pitch_range[1] - profile.pitch_range[0]:.1f} Hz)
            - Pauses: {profile.average_pause_duration:.3f} seconds on average
            - Emotion: {profile.emotion_profile.get('dominant_emotion')} (confidence: {profile.emotion_profile.get('confidence'):.2f})
            """
        
        report_path = output_dir / "comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"   Report saved: {report_path}")
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        print(f"\nOutput files in: {output_dir}")
        print(f"  - original.wav (default Piper voice)")
        print(f"  - personalized.wav ({profile.user_name}'s voice)")
        print(f"  - comparison_report.txt (detailed analysis)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}", exc_info=True)
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Piper TTS Personalization Engine - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Create a voice profile
        python cli.py create-profile --user-id john --user-name "John Doe" --audio my_voice.wav

        # Synthesize speech with raw text
        python cli.py synthesize --user-id john --text "Hello world" --output output.wav
        
        # Synthesize speech from a text file
        python cli.py synthesize --user-id john --text_file input.txt --output output.wav

        # Compare original vs personalized
        python cli.py compare --user-id john --text_file input.txt --output-dir ./comparison
                """
            )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # create-profile command
    create_parser = subparsers.add_parser('create-profile', help='Create a voice profile from audio')
    create_parser.add_argument('--user-id', required=True, help='Unique user ID')
    create_parser.add_argument('--user-name', required=True, help='User display name')
    create_parser.add_argument('--audio', required=True, help='Path to audio file (WAV, MP3, FLAC)')
    create_parser.set_defaults(func=cmd_create_profile)
    
    # synthesize command
    synth_parser = subparsers.add_parser('synthesize', help='Synthesize text with personalized profile')
    synth_parser.add_argument('--user-id', required=True, help='User ID (profile must exist)')
    synth_parser.add_argument('--output', default='output.wav', help='Output audio file path')
    synth_parser.add_argument('--model', default=None, help='Path to Piper .onnx model file (optional)')
    
    synth_input_group = synth_parser.add_mutually_exclusive_group(required=True)
    synth_input_group.add_argument('--text', type=str, help='Text to synthesize')
    synth_input_group.add_argument('--text_file', type=str, help='Path to a .txt file containing text')
    synth_parser.set_defaults(func=cmd_synthesize)

    
    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare original vs personalized synthesis')
    compare_parser.add_argument('--user-id', required=True, help='User ID (profile must exist)')
    compare_parser.add_argument('--output-dir', default='./comparison', help='Output directory for comparison files')
    
    compare_input_group = compare_parser.add_mutually_exclusive_group(required=True)
    compare_input_group.add_argument('--text', type=str, help='Text to use for comparison')
    compare_input_group.add_argument('--text_file', type=str, help='Path to a .txt file containing text')
    compare_parser.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    # Execute command
    try:
        success = args.func(args)
        return success
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)