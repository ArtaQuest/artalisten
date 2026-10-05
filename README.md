# ArtaListen

ArtaListen transcribes and translates long recordings made in extreme noise at a very low signal-to-noise ratio. It is a local Python program. It does not call a paid API. Weights, recordings, and transcripts stay on the machine that runs it.

The first target is a distant cafe microphone: two people talking in Russian, one man and one woman, with an English song in the background.

Low-SNR recovery details for the Anna cafe passes (forced Russian, junk filter, consensus, Anna-first stems) are in [METHOD.md](METHOD.md) (pass5 / v5).

## Install

Python 3.11 or newer, and `ffmpeg`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install -r requirements.txt` installs the same runtime libraries without the command entry point. `pip install -e ".[diarize]"` also installs `pyannote.audio` for the gated diarizer. That extra is unused unless a Hugging Face token is already on the machine.

## Run

### Pass5 recovery (Mac-local, recommended for Voice Memos)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# single file
caffeinate -i artalisten recover ~/Downloads/'Voice 261005_104834.m4a' \
  --profile profiles/anna.json \
  --deliverables deliverables/memo-local/

# batch Voice Memos from Downloads
caffeinate -i artalisten memo --inbox ~/Downloads --glob 'Voice*.m4a' \
  --profile profiles/anna.json \
  --deliverables deliverables/memo-local/
```

See [METHOD.md](METHOD.md) for the pass5 / v5 method. Kaggle scripts under `kaggle/`
remain an optional remote GPU fallback; they are not required.

### Legacy single-path


```bash
artalisten process ~/Downloads/anna+brother.m4a --translate en
```

Outputs land in `./runs/anna-brother/` (the `+` in the file name is replaced so the directory is easy to type):

| File | What it is |
| --- | --- |
| `vocals.wav` | Speech stem after music separation |
| `enhanced.wav` | Speech stem after denoising |
| `speech_16k.wav` | 16 kHz audio used for speakers, transcript, and translation |
| `diarization.json` | Global speaker profiles and the timeline |
| `transcript.md` | Timestamped Russian transcript and English translation |
| `utterances.json` | The same turns, including word timestamps |
| `manifest.json` | Models, devices, and how much of the file was processed |

A finished run prints the path to `transcript.md`. Stages that already have outputs are skipped. `--force` recomputes them. `--max-seconds 120` processes a prefix; speaker profiles are then learned on that prefix only, and the transcript says so.

`--translate` accepts `en` only. Translation is the Whisper translate task.

## Pipeline

Each stage finishes and unloads its model before the next one starts. Separation, enhancement, and speaker embeddings try MPS when `torch.backends.mps` is available, and fall back to CPU if that op fails. faster-whisper does not use MPS: CTranslate2 on this Mac runs int8 on CPU.

### 1. Separate the song from the speech

Model: **HTDemucs** (`htdemucs` in Demucs).

The cafe recording is speech plus a full song. HTDemucs splits a mix into drums, bass, other, and vocals. ArtaListen keeps the vocals stem and records a coarse energy envelope of the accompaniment. The English singing is itself vocal, so some lyric bleed remains in this stem. Later stages flag that bleed instead of pretending the separator removed it.

Default `--shifts 0` skips Demucs' extra shifted passes so a long file can finish on a laptop. Audio is split into the model's native segments (`split=True`) so the whole half-hour is not one activation.

### 2. Enhance speech at very low SNR

Model: **DeepFilterNet3**, with its post-filter when this version of the package accepts it.

The vocal stem is still a distant microphone in a noisy room. DeepFilterNet3 is a small open speech enhancer built for low SNR. It runs in 20 second chunks with a one second crossfade so a half-hour file does not become one spectrogram.

### 3. Learn speakers from the whole file

This happens before any transcript is written.

A diarizer that clusters each chunk on its own assigns arbitrary ids inside that chunk. The same voice becomes "speaker 0" for five minutes and "speaker 1" for the next five. ArtaListen does not do that.

1. **Silero VAD** finds speech on the enhanced file.
2. **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) embeds every speech window of about two seconds.
3. Those embeddings, from the entire processed recording, are clustered **once** with average-linkage cosine clustering. The default prior is **2 speakers**.
4. Median fundamental frequency is only a weak hint for naming the two profiles. The lower pitch is labelled `man (estimated)` and the higher pitch `woman (estimated)`. If the two pitches are within 15 Hz, or either one is missing, the labels stay `speaker_N (estimated, sex uncertain)`.

If `HF_TOKEN` (or a Hugging Face token file) is set and `pyannote.audio` is installed, the speaker stage uses **pyannote/speaker-diarization-3.1** on the full file instead, still with the two-speaker prior and the same estimated pitch names. This machine's open path does not need that token. Accepting the pyannote model licence is a manual step on the Hugging Face account; ArtaListen will not ask you to do it.

### 4. Transcribe

Model: **faster-whisper `large-v3`**, compute type **int8**, language **Russian**, word timestamps on.

The enhanced 16 kHz audio is transcribed in five-minute chunks with a short overlap so a long job can resume. Chunking is only for the recognizer. Speaker names on each word come from the global timeline built in stage 3, not from a fresh clustering of that chunk. Words are merged into turns when the speaker stays the same and the gap is under 0.8 seconds.

A turn whose letters are mostly Latin, or that piles up common English lyric words, is marked **likely English lyric bleed**. The line stays in the transcript.

### 5. Translate to English

The same large-v3 model runs Whisper's **translate** task, which writes English. Turns stay speaker-attributed: an English word is attached to a Russian turn only when it falls in that turn and carries the same global speaker label.

**SeamlessM4T v2 is not used.** A MacBook Air with 16 GB of unified memory does not leave a safe margin for SeamlessM4T v2 large beside the system and large-v3. Whisper's translate task reuses the weights already required for the transcript, so there is no second multi-gigabyte translation model.

## Cafe recording

```bash
artalisten process ~/Downloads/anna+brother.m4a --translate en
```

That file is a distant mono microphone in a public cafe, about 34 minutes, one Russian man and one Russian woman, English music in the background. Expect the run to take a long time on CPU. `caffeinate -i` in front of the command stops the laptop sleeping mid-run:

```bash
caffeinate -i artalisten process ~/Downloads/anna+brother.m4a --translate en
```

Do not commit the recording or the transcript. Both are gitignored, along with `.venv/` and downloaded weights.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover global clustering, pitch labels, lyric flags, and the command help. They do not download models.

## Models and licences

| Stage | Model | Licence to check |
| --- | --- | --- |
| Separation | Demucs HTDemucs (`htdemucs`) | MIT |
| Enhancement | DeepFilterNet3 | MIT / Apache-2.0 (see the DeepFilterNet repo) |
| Voice activity | Silero VAD | MIT |
| Speaker embeddings | SpeechBrain ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 |
| Optional diarization | pyannote/speaker-diarization-3.1 | gated; requires a Hugging Face token and accepted terms |
| Transcript and translation | faster-whisper `large-v3` (OpenAI Whisper weights via CTranslate2) | MIT |

Weights download into the Hugging Face cache and `~/.cache/artalisten` on first use.
