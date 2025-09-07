uv run python pytorch-CycleGAN/test.py \
--dataroot path-to-your-images \
--name histology_stain_cyclegan_training \
--model cycle_gan \
--checkpoints_dir checkpoints \
--results_dir path-to-your-results-folder-you-want-to-save
# --continue_train