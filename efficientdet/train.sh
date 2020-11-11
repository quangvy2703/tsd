python main.py --mode=train \
    --training_file_pattern=tfrecord/train*.tfrecord \
    --model_name=efficientdet-d2 \
    --model_dir="/content/drive/My Drive/models/efficientdet-d2-finetune"  \
    --ckpt=checkpoints/efficientdet-d2  \
    --train_batch_size=4 \
    --eval_batch_size=4 --eval_samples=1024 \
    --num_examples_per_epoch=4500 --num_epochs=100  \
    --hparams=configs/traffic_config.yaml