# Scripts that help with debugging

## Transcribe alignments using Whisper
Once you have the alignments (i.e., after step 5.4), you can use this script to transcribe the alignments and also listen to the alignments.

Install Whisper by `pip install -U openai-whisper`.
```shell
python -m analysis.dump_alignments \
    --align_path example/voxpopuli/alignments/en-de/20180313-0900-PLENARY-15_en-20180313-0900-PLENARY-15_de.txt \
    --src_segs example/voxpopuli/segments/en/20180313-0900-PLENARY-15_en.txt \
    --src_wav example/voxpopuli/raw_audios/en/20180313-0900-PLENARY-15_en.ogg \
    --tgt_segs example/voxpopuli/segments/de/20180313-0900-PLENARY-15_de.txt \
    --tgt_wav example/voxpopuli/raw_audios/de/20180313-0900-PLENARY-15_de.ogg \
    --out_dir ./outputs \
    --asr \
    --src_lang en --tgt_lang de \
    --whisper_size medium --whisper_root ./whisper_cache
```
The output will be some html files, which you can open with your browser.
