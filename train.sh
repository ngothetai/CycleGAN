uv run python pytorch-CycleGAN-and-pix2pix/train.py \
--dataroot ./datasets/histology_stain \
--name histology_stain_cyclegan \
--model cycle_gan \
--batch_size 8 \
--load_size 256 \
--crop_size 256 \
--n_epochs 100 \
--use_wandb
# --continue_train