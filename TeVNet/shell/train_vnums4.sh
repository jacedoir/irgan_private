CUDA_VISIBLE_DEVICES=0 \
python train.py \
--smp_model Unet --smp_encoder resnet18 --smp_encoder_weights imagenet \
--num-epochs 200 \
--num-epochs-save 10 \
--num-epochs-val 10 \
--resume /w/340/murdock/irgan/tev-out/KAIST/epoch_60.pth \
--outputs-dir  /w/340/murdock/irgan/tev-out/KAIST \
--batch-size 32 \
--lr 0.001 \
--start_epoch 60 \
--train-dir /w/340/murdock/irgan/datasets/KAIST/train \
--eval-dir /w/340/murdock/irgan/datasets/KAIST/train \
--vnums 4