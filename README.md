# FastSpeech 2 Text-to-Speech: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models, built from the public
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) PyTorch code
(paper: *FastSpeech 2*, Ren et al.). Each language has its own folder with its own
`preprocess.py`, `train.py`, `inference.py` and `config.yaml`.

```
English/   Svarah, 102 speaker groups, CMU/LibriSpeech lexicon + g2p_en, MFA english_us_arpa
Telugu/    IndicTTS Telugu, 2 speakers, rule-based Telugu G2P, MFA trained on the data
Results/   loss plots, test audios, variance-adaptor plots (step 8000 of each model)
Report/    Technical_Report.pdf
vocoder/   pretrained HiFi-GAN (not in git, see vocoder/README.md)
plot_losses.py   turns the loss CSV logs into the two loss plots
```

## Model checkpoints (400 MB each, too big for git)

| Model | Step | Download |
|---|---|---|
| English | 8000 | https://drive.google.com/file/d/19J7YtJFVAvZvGlByPBqXUljWNob-rkD_/view?usp=sharing (save as `English/checkpoints/8000.pth.tar`) |
| Telugu | 8000 | https://drive.google.com/file/d/1YGv0a4xW0NCj8Hfu_DrJvrxwaakLXftw/view?usp=sharing (save as `Telugu/checkpoints/8000.pth.tar`) |

## Results at a glance

| | English | Telugu |
|---|---|---|
| Data (train / validation) | 4.92 h / 1.01 h (3,257 / 649 clips) | 5.07 h / 1.07 h (2,058 / 476 clips) |
| After alignment (used for training) | 3,249 / 648 utterances, 4.55 h | 2,048 / 463 utterances, 4.22 h |
| Speakers | 102 metadata groups | 2 (TEF female, TEM male) |
| Training | 8,000 steps, batch 16, Tesla T4 (about 78 min) | 8,000 steps, batch 16, Tesla T4 (about 1.5 h) |
| Final training loss | 1.157 | 0.989 |
| Final / best validation loss | 3.451 / 3.292 (step 1,500) | 3.943 / 3.663 (step 1,500) |
| Test speaker | S073 (most training data) | TEF |

Both models overfit: the training loss keeps falling while the validation loss is best very early
(see `Results/*/training_loss.png` and `validation_loss.png`; the numbers come from `Results/*/logs/*.csv`). The Telugu audio has clear harmonic
structure; the English audio is noisy and not clearly intelligible (see the report, Section 4).

## Run inference with the released checkpoints

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['averaged_perceptron_tagger_eng','averaged_perceptron_tagger','cmudict','punkt']]"
mkdir -p vocoder && unzip generator_universal.pth.tar.zip -d vocoder/      # see vocoder/README.md
cd English && python inference.py config.yaml --restore_step 8000 --out_dir ../Results/English
cd ../Telugu && python inference.py config.yaml --restore_step 8000 --out_dir ../Results/Telugu
```
`English/preprocessed_data/` and `Telugu/preprocessed_data/` contain `stats.json`, `speakers.json`,
`train.txt` and `val.txt`, which inference needs.

## What was changed compared with the reference code

Both languages
- Single `config.yaml` per language (replaces the three YAML files).
- `utils/model.py`, `audio/*`: updated for current PyTorch (`weights_only`) and librosa >= 0.10; the hard-coded `.cuda()` calls were removed so preprocessing also runs on CPU.
- `preprocessor/preprocessor.py`: speaker folders read in sorted order; missing or broken TextGrids are skipped (the original reused variables of the previous utterance); MFA 3.x silence intervals (empty text) mapped to `sp`; UTF-8 text files; train/validation lists taken from a split CSV so that **no recording overlaps** between the sets; optional `fill_gaps` (see Known issues).
- `train.py`: loss values saved to CSV (for plots), `--resume` to continue after a disconnect, old checkpoints pruned, training does not stop if a TensorBoard sample fails.
- `synthesize.py` became `inference.py`: reads `test_sentences.json`, writes one wav per sentence and the predicted **duration, pitch (F0) and energy plots** (plus a JSON with the numbers).
- Mandarin / pinyin / AISHELL / LJSpeech / LibriTTS code removed.

English only: `frontend.py` expands numbers, times and dates (`10:45 AM` becomes "ten forty five ay em", `12 November 2026` becomes "twelve November twenty twenty six"), applies a noun/verb rule for the heteronym *conduct* (E2a / E2b), and uses `g2p_en` for unseen words such as *andaroo*.

Telugu only: `text/telugu.py` (rule-based Telugu G2P with 49 phones and number/ordinal/time expansion), `text/symbols.py`, `text/cleaners.py`, `make_mfa_dictionary.py`, and `frontend.py`.

## Reproducing the results

Data selection produced `svarah_split.csv` and `telugu_split.csv` (train/validation assignment). Copy them to
`English/data_split/` and `Telugu/data_split/`, and put the audio (`<speaker>/<utt>.wav` + `.lab`, 22.05 kHz mono) in `raw_data/`.

```bash
# Forced alignment (MFA 3.4.3, separate conda environment)
mfa model download acoustic english_us_arpa && mfa model download dictionary english_us_arpa && mfa model download g2p english_us_arpa
mfa align raw_data english_us_arpa english_us_arpa aligned_en --clean -j 4 --g2p_model_path english_us_arpa
python Telugu/make_mfa_dictionary.py raw_data_te telugu.dict
mfa train raw_data_te telugu.dict telugu_acoustic.zip --output_directory aligned_te -c mono.yaml --clean -j 2   # mono.yaml: monophone only, 20 iterations
# copy the TextGrids to <language>/preprocessed_data/TextGrid/<speaker>/

cd English            # or Telugu
python preprocess.py config.yaml
python train.py config.yaml            # add --resume to continue after a disconnect
python inference.py config.yaml --restore_step 8000 --out_dir ../Results/English
cd .. && python plot_losses.py --log_dir English/output/log --out_dir Results/English
```

## Known issues and limitations (also discussed in the report)

- Svarah is an ASR corpus (noisy, many speakers, about 3 minutes each); the English audio is not clearly intelligible.
- Only 8,000 training steps were possible in the available time; validation loss is best long before that (overfitting).
- **Possible alignment-gap issue.** `train.txt` of the English model contains no `sp` token at all and preprocessing kept only 4.55 of 5.93 hours (Telugu 4.22 of 6.14 hours). A likely reason is that MFA leaves silences out of the TextGrid as gaps, so the durations do not cover the whole audio segment. `preprocessing.fill_gaps: True` in `config.yaml` handles this (silences of at least 60 ms become `sp`, shorter gaps are absorbed). The submitted results were produced with `fill_gaps: False`; there was no time to retrain.
- Telugu durations come from a monophone-only MFA model (a longer 5-stage run was lost when the session ended).
- The vocoder (universal HiFi-GAN) is not fine-tuned for either language.
- Telugu number expansion is rule-based and was not checked by a native speaker; Telugu homographs with identical spelling (T2a / T2b *పట్టు*) get identical phonemes.
