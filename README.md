# Audio Transcription Pipeline

Converts an audio file into text with per-segment timestamps. Built with Python, ffmpeg and faster-whisper (open-source Whisper).

## Quick start

Requirements: Python 3.9+ and ffmpeg on PATH (`ffmpeg -version`).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python transcribe_pipeline.py test_audio/speech_short.wav --backend mock          # no model, instant
python transcribe_pipeline.py test_audio/speech_short.wav --model base            # real transcription
python transcribe_pipeline.py test_audio/long_speech.mp3 --model tiny -o long.json --srt long.srt
```

The first real run downloads the model (base is about 150 MB). The HF_TOKEN warning is harmless.

Options: `--model tiny|base|small|medium|large-v3`, `--language ur|pa|en|auto`, `--max-chunk SECONDS` (min 10), `--workers N`, `-o out.json`, `--srt out.srt`, `--backend whisper|mock`.

## Output

```json
{
  "source": "speech_short.wav",
  "duration": 10.06,
  "num_chunks": 1,
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.6, "text": "Hello, this is a test on the Transcription pipeline."}
  ],
  "text": "..."
}
```

Timestamps are absolute relative to the original file, even when the audio was chunked. SRT export is optional.

## How it works

```
input file -> normalize (ffmpeg) -> chunk at silences -> ASR backend per chunk -> offset timestamps -> postprocess -> JSON / SRT
```

| Stage | Function | Purpose |
|---|---|---|
| Validate + normalize | `normalize()` | Any format to 16 kHz mono 16-bit WAV |
| Chunk | `detect_silences()`, `plan_cuts()`, `make_chunks()` | Bounded chunks cut at pauses |
| Transcribe | `WhisperBackend`, `MockBackend` | Swappable engine behind a `Backend` interface |
| Merge | `transcribe_file()` | Shift chunk times by offset, keep order |
| Clean up | `postprocess()` | Whitespace, empty and duplicate segments, fragment merging |
| Output | `to_srt()`, `main()` | JSON and SRT |

## Design decisions (Part 1)

**Why faster-whisper.** Open source, strong multilingual accuracy, native segment timestamps, int8 on CPU (no GPU needed), built-in VAD, about 4x faster than openai-whisper. The engine is behind a one-method interface, so a cloud API, WhisperX or Vosk is a small class.

**Audio formats.** Never branch on file extension. ffmpeg decodes everything (WAV, MP3, M4A, FLAC, OGG, MP4 video) to 16 kHz mono PCM WAV, so one code path serves every input and the model always sees the same shape. Audio is passed to Whisper as a numpy array, so a PyAV version mismatch cannot break the pipeline.

**Long audio.** Files over 10 min are split into chunks of about 5 min (60 s min, 10 min max), cut at the silence nearest the target so words are rarely sliced; if no silence exists, hard-cut at the max. A final chunk under 1 s is merged into the previous one. Each chunk is transcribed independently and its timestamps shifted by its offset. Memory stays flat regardless of file length, and a failed chunk can be retried alone.

**Input safety.** Missing file, directory, empty file, corrupt or non-audio file, no audio stream, ffmpeg not installed and invalid `--max-chunk` each give a one-line error and exit code 1.

**Postprocessing.** Whitespace normalization, empty-segment removal, collapsing near-adjacent identical segments (Whisper hallucination loops), merging sentence fragments.

**Concurrency in the script.** `--workers` only applies to thread-safe backends (mock, remote API). The local Whisper backend is forced to 1 worker.

## System design (Part 2, not implemented)

### Concurrent uploads
- API servers stay stateless and never run ASR in the request path. The client uploads straight to object storage via a presigned URL (large files bypass the API), then calls the API to register the job.
- The job goes onto a queue (SQS, RabbitMQ or Redis streams). Separate GPU/CPU workers pull from the queue and scale on queue depth, not on HTTP traffic.
- Backpressure and fairness: per-user rate limits and quotas, max file size and duration checks, priority queues so short files are not blocked behind long ones.
- Chunk-level parallelism: a long file fans out into chunk tasks processed by many workers, then a merge step reassembles them.

### Storage
- **Audio:** object storage (S3/GCS/MinIO), key `audio/{user}/{job_id}/original.ext`, with a lifecycle policy (delete or move to cold storage after N days), encryption at rest, and the normalized WAV and chunks treated as disposable cache.
- **Transcripts:** the full JSON also goes to object storage, since it can be large. Metadata and segments go to PostgreSQL: `jobs(id, user, status, attempts, error, audio_uri, created_at...)` and `segments(job_id, idx, start, end, text)`, with full-text search (or OpenSearch) on text if search is needed.
- Retention and privacy: configurable retention, delete-on-request, PII awareness, signed short-lived URLs for access.

### Retries and recovery
- **Job state machine:** `queued -> processing -> completed | failed`, stored in the database as the source of truth.
- **Retries:** automatic retry with exponential backoff and jitter for transient errors (network, worker crash, model OOM). Permanent errors (corrupt file, unsupported format) fail fast, no retry.
- **Chunk-level checkpointing:** each finished chunk's segments are saved immediately, so a retry resumes from the failed chunk instead of restarting the whole file.
- **Idempotency:** deterministic chunk IDs (`job_id + chunk_index`) and upserts mean duplicate deliveries are harmless.
- **Visibility timeout and heartbeats:** if a worker dies, the message reappears and another worker picks it up.
- **Dead-letter queue** after N attempts, with alerting. Failed jobs can be re-driven manually.
- **Fallbacks:** retry with a smaller model or another backend on repeated failure. Monitor failure rate, queue age and processing time per audio minute.

### API
```
POST /v1/uploads                -> { upload_url, upload_id }       (presigned URL)
POST /v1/transcriptions         { upload_id, language?, model? }   -> 202 { job_id, status }
GET  /v1/transcriptions/{id}    -> { status, progress, error? }
GET  /v1/transcriptions/{id}/result?format=json|srt|txt
DELETE /v1/transcriptions/{id}
```
- Asynchronous by design: 202 Accepted plus polling, with optional webhook callback or SSE for progress. Long audio would time out a synchronous request.
- Idempotency-Key header on POST to prevent duplicate jobs, API-key or OAuth auth, rate limiting, validation (type, size, duration), versioned paths, OpenAPI docs.
- Implementation: FastAPI wrapping `transcribe_file()` in a worker (Celery/RQ), with Postgres and S3 behind it.

## Limitations and next steps
- A hard cut in continuous speech can split a word. Fix: overlap plus timestamp dedupe, or VAD-based chunking.
- No word-level timestamps or speaker diarization (WhisperX, pyannote).
- Accuracy depends on the model size. Use `small` or larger for non-English audio and pass `--language`.
- Add automated tests and word-error-rate benchmarks.

## Testing performed
- WAV, MP3, M4A and FLAC give matching duration and structure.
- A 15.5 min file splits into 3 chunks, and into 50 chunks with `--max-chunk 30`; timestamps stay ordered and end at 930.4 s.
- Edge cases: missing, empty, directory, corrupt and non-audio input; silent audio; MP4 with and without audio; filenames with spaces, accents and a leading dash; ffmpeg missing; invalid chunk size.
- The Whisper path was verified with a stub model (numpy input) and with the real `base` model on short files.

## Repository layout
```
transcribe_pipeline.py   # the pipeline
requirements.txt
README.md
test_audio/              # synthetic speech samples in several formats + a 15 min file
```