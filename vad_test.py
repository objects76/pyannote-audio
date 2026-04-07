"""VAD minimal test with 촉법소년.m4a"""

from pathlib import Path

from load_audio import load_audio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

AUDIO_FILE = Path(__file__).parent / "촉법소년.m4a"


def test_vad():
    assert AUDIO_FILE.exists(), f"Audio file not found: {AUDIO_FILE}"

    weight_dir = Path(
        "/home/jjkim/Desktop/work/Diarization/rsup-gitlab/weight-for-spk-diar/pyan30"
    )
    checkpoint = weight_dir / "pytorch_model.bin"
    assert checkpoint.exists()

    print(f"[TRY] Loading model from {weight_dir.name} ...")
    model = Model.from_pretrained(checkpoint, strict=False)
    print(f"[OK] Model loaded: {model.__class__.__name__}")

    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.instantiate(
        {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
    )

    print(f"\n[RUN] Processing {AUDIO_FILE.name} ...")

    waveform = load_audio(AUDIO_FILE)  # (1, num_samples), float32, 16kHz mono
    sample_rate = 16000

    vad_result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    print(f"\n[RESULT] VAD output:")
    for segment, _, label in vad_result.itertracks(yield_label=True):
        print(f"  {segment.start:7.2f}s - {segment.end:7.2f}s  [{label}]")

    total_speech = sum(s.duration for s in vad_result.get_timeline())
    print(f"\n  Total speech: {total_speech:.2f}s")
    print(f"  Segments: {len(vad_result)}")
    assert len(vad_result) > 0, "No speech segments detected!"
    print("\n[PASS] VAD test passed!")


if __name__ == "__main__":
    test_vad()
