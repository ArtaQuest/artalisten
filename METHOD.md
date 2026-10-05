# Recovery method — pass5 / v5

ArtaListen pass5 targets the failure modes seen on the 2026-10-05 Anna cafe Voice
files and the earlier soft-voice cafe train/test pair.

## Failure modes this pass addresses

- Full-context Whisper inventing subtitle credits («Субтитры… DimaTorzok», «Симон»),
  Ukrainian «Дякую за перегляд», «Продолжение следует», music captions
  («Играет музыка», «Девушки отдыхают»), and English lyric/translate loops.
- Soft Anna drowned by a louder man or by music.
- Auto language detection flipping to `uk` / `nn` / `en` and emitting junk captions.
- Cross-window conditioning that turns one bad token into a repetition loop.
- Treating a single cleaning variant’s phrase as recovered dialogue.

## Upgrades

1. **Forced Russian** — cafe / Anna jobs always pass `language="ru"`. Auto-detect is refused.
2. **Stronger junk filter** (`artalisten.junk`) — credits, captions, viewer-thanks, StSq tags,
   and known translate loops are rejected before scoring or consensus.
3. **Anna-first enhancement** — when an Anna ECAPA profile is present, prefer the cluster with
   higher cosine-to-Anna (else higher-F0 / majority); apply ~140–6500 Hz band and lift her mask;
   attenuate non-Anna / gaps.
4. **Decode strategy** — short independent windows (10 s, hop 8 s) with
   `condition_on_previous_text=False`, plus one full-context pass for comparison. Score by
   non-junk words at probability ≥ 0.4.
5. **Consensus gate** (`artalisten.consensus`) — a phrase enters Final only when ≥2 independent
   cleaning variants agree. One-decode-only hits are marked weak.
6. **Cleaning variants** — A demucs-only, B demucs+sepformer, E demucs+Anna-band+level,
   F unseparated+band, G demucs+Anna-band+dynnorm (ultra-quiet).
7. **Cafe two-speaker** — fit speakers only on `anna+brother.m4a`; freeze centroids for
   `anna+brother_test.m4a`. Never fit on the held-out test file.
8. **Translate** — Whisper translate runs on the same audio; English lyric/junk lines are
   stripped and never pasted onto Russian turns.

## Code map

| Piece | Where |
| --- | --- |
| Junk filter | `src/artalisten/junk.py` |
| Consensus | `src/artalisten/consensus.py` |
| Independent windows | `src/artalisten/asr.py` → `transcribe_independent_windows` |
| Anna band / dynnorm / mask | `src/artalisten/enhance.py` |
| Cafe soft-voice kernel | `kaggle/run.py` |
| Multi-variant voice kernel | `kaggle/pass5_run.py` |

## Local tests

```bash
pip install -e ".[dev]"
pytest
```
