Test files (synthetic speech from espeak-ng, robotic voice but Whisper handles it fine)

speech_short.wav/.mp3/.m4a/.flac  10s, same sentence in 4 formats -> tests format handling
speech_second.wav                 18s, longer sentences
long_speech.mp3                   ~15.5 min -> exceeds 10 min, triggers chunking (3 chunks)

Expected text of speech_short:
 "Hello, this is a test of the transcription pipeline. The quick brown fox jumps over the lazy dog.
  Today is a good day to build something useful."

Run:
  python transcribe_pipeline.py test_audio/speech_short.wav --model base
  python transcribe_pipeline.py test_audio/speech_short.mp3 --model base      # same result expected
  python transcribe_pipeline.py test_audio/long_speech.mp3 --model base --srt long.srt   (slow on CPU; use --model tiny)
  python transcribe_pipeline.py test_audio/long_speech.mp3 --backend mock     # instant, plumbing only
