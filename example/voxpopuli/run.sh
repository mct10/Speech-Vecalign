#!/bin/bash

# An example script to run all steps

metadata=./example/voxpopuli/metadata.tsv
src_lang=en
tgt_lang=de
speech_laser_dir=$1  # set it yourself
src_laser=english.pt
tgt_laser=germanic.pt
out_dir=./outputs

seg_dir=${out_dir}/segments
untrans_seg_dir=${out_dir}/untrans_segs
cat_seg_dir=${out_dir}/cat_segs
untrans_cat_seg_dir=${out_dir}/untrans_cat_seg_ids
embed_dir=${out_dir}/embeds
align_dir=${out_dir}/alignments

set -x
# preprocess steps
# 4.1
python -m svecalign.preprocess.segment \
    ${metadata} ${seg_dir} \
    --lang ${src_lang} \
    --vad_version snakers4/silero-vad:v4.0 \
    --cache_dir ./vad_cache

python -m svecalign.preprocess.segment \
    ${metadata} ${seg_dir} \
    --lang ${tgt_lang} \
    --use_tgt \
    --vad_version snakers4/silero-vad:v4.0

# 4.2
python -m svecalign.preprocess.detect_untranslate_segs \
    ${metadata} ${untrans_seg_dir} \
    --seg_dir ${seg_dir} \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang}

# alignment steps
# 5.1
python -m svecalign.seg_align.concat_segs \
    ${metadata} ${cat_seg_dir} \
    --seg_dir ${seg_dir} \
    --lang ${src_lang}
python -m svecalign.seg_align.concat_segs \
    ${metadata} ${cat_seg_dir} \
    --seg_dir ${seg_dir} \
    --lang ${tgt_lang} \
    --use_tgt

# 5.2
python -m svecalign.seg_align.detect_untranslate_concats \
    ${metadata} ${untrans_cat_seg_dir} \
    --seg_dir ${seg_dir} \
    --identical_seg_dir ${untrans_seg_dir} \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang}

# 5.3
python -m svecalign.seg_align.embed \
    ${metadata} ${embed_dir} \
    --concat_dir ${cat_seg_dir} \
    --lang ${src_lang} \
    --embed_model_type speech_laser \
    --sl_ckpt_dir ${speech_laser_dir} --sl_ckpt_name ${src_laser}

python -m svecalign.seg_align.embed \
    ${metadata} ${embed_dir} \
    --concat_dir ${cat_seg_dir} \
    --lang ${tgt_lang} \
    --embed_model_type speech_laser \
    --sl_ckpt_dir ${speech_laser_dir} --sl_ckpt_name ${tgt_laser} \
    --use_tgt

# 5.4
python -m svecalign.seg_align.align \
    ${metadata} ${align_dir} \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --seg_dir ${seg_dir} \
    --concat_dir ${cat_seg_dir} \
    --embed_dir ${embed_dir} \
    --is_stopes_embed \
    -a 6 \
    --ign_indices_dir ${untrans_cat_seg_dir}

# postprocessing steps
# 6.1
python -m svecalign.postprocess.filter_by_cost \
    ${metadata} ${align_dir}_0.7 \
    --align_dir ${align_dir} \
    --max_cost 0.7 \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang}

# 6.2
python -m svecalign.postprocess.filter_untrans_align \
    ${metadata} ${align_dir}_0.7_clean \
    --align_dir ${align_dir}_0.7 \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --seg_dir ${seg_dir} \
    --n_proc 1 --save_audio

# 6.3
python -m svecalign.postprocess.concat_aligns \
    ${metadata} ${align_dir}_0.7_clean_cat3 \
    --max_num_align 3 \
    --align_dir ${align_dir}_0.7_clean \
    --seg_dir ${seg_dir} \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --apply_dur_cond_to_both_sides \
    --max_dur 20.0

# 6.4
python -m svecalign.postprocess.filter_by_dur \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s \
    --align_dir ${align_dir}_0.7_clean_cat3 \
    --seg_dir ${seg_dir} \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --min_dur 1.0

# 6.5
python -m svecalign.postprocess.embed_align \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_embed \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --align_dir ${align_dir}_0.7_clean_cat3_min1s \
    --seg_dir ${seg_dir} \
    --concat_seg_dir ${cat_seg_dir} \
    --concat_seg_embed_dir ${embed_dir} \
    --embed_model_type speech_laser \
    --sl_ckpt_dir ${speech_laser_dir} --sl_ckpt_name ${src_laser}

python -m svecalign.postprocess.embed_align \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_embed \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --align_dir ${align_dir}_0.7_clean_cat3_min1s \
    --seg_dir ${seg_dir} \
    --concat_seg_dir ${cat_seg_dir} \
    --concat_seg_embed_dir ${embed_dir} \
    --embed_model_type speech_laser \
    --sl_ckpt_dir ${speech_laser_dir} --sl_ckpt_name ${tgt_laser} \
    --use_tgt

# 6.6
python -m svecalign.postprocess.prep_index \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_embed_indexes \
    --data_dir ${align_dir}_0.7_clean_cat3_min1s_embed \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --embed_fp16 \
    --sample_ratio 0.5 --embed_stopes

python -m svecalign.postprocess.prep_index \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_embed_indexes \
    --data_dir ${align_dir}_0.7_clean_cat3_min1s_embed \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --embed_fp16 \
    --sample_ratio 0.5 --embed_stopes \
    --use_tgt

# 6.7
python -m svecalign.postprocess.score_align \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_margin \
    --embed_dir ${align_dir}_0.7_clean_cat3_min1s_embed \
    --align_dir ${align_dir}_0.7_clean_cat3_min1s \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --index_dir ${align_dir}_0.7_clean_cat3_min1s_embed_indexes \
    --embed_fp16 \
    --embed_stopes

# 6.8
python -m svecalign.postprocess.prep_tsv \
    ${metadata} ${align_dir}_0.7_clean_cat3_min1s_tsvs \
    --src_lang ${src_lang} --tgt_lang ${tgt_lang} \
    --align_dir ${align_dir}_0.7_clean_cat3_min1s_margin \
    --seg_dir ${seg_dir}

# 6.9
python -m svecalign.postprocess.remove_overlaps \
    --output_dir ${align_dir}_0.7_clean_cat3_min1s_tsvs/${src_lang}-${tgt_lang} \
    --output_filename align.rm_overlap.tsv.gz \
    --mining_result_path ${align_dir}_0.7_clean_cat3_min1s_tsvs/${src_lang}-${tgt_lang}/align.tsv.gz \
    --min_audio_length 2000 \
    --mining_threshold 0.0 \
    --max_overlap 0.8

# 6.10
python -m svecalign.postprocess.sort_tsv \
    --in_tsv ${align_dir}_0.7_clean_cat3_min1s_tsvs/${src_lang}-${tgt_lang}/align.rm_overlap.tsv.gz \
    --out_tsv ${align_dir}_0.7_clean_cat3_min1s_tsvs/${src_lang}-${tgt_lang}/align.rm_overlap.sort.tsv.gz
