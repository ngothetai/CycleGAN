uv run python pytorch-CycleGAN/train.py \
--dataroot ./datasets/histology_stain \
--name histology_stain_cyclegan_training \
--model cycle_gan \
--batch_size 8 \
--lambda_A 5.0 \
--lambda_B 5.0 \
--lambda_identity 0.1 \
--lr 0.0003 \
--load_size 256 \
--crop_size 256 \
--n_epochs 10 \
--save_epoch_freq 1 \
--use_wandb
# --continue_train