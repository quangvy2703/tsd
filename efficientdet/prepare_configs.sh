export MODEL=efficientdet-d0
export DATA_PATH='/home/ubuntu/Downloads/Compressed/za_traffic_2020'
export DATA_ANNOTATIONS_PATH='/home/ubuntu/Downloads/Compressed/za_traffic_2020/annotations/instances_train2017.json'

# extend data
export USE_EXTEND_DATA=1
export EXTEND_DATA_PATH='/home/ubuntu/papers/TSD/automl/automl/efficientdet/data/cure'
export DATA_EXTEND_ANNOTATIONS_PATH='/home/ubuntu/papers/TSD/automl/automl/efficientdet/data/cure/annotations/cure_annotations.json'
cd automl/efficientdet/

#pip install -r requirements.txt


# Download backbone
mkdir checkpoints
cd checkpoints
#wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
#tar xf ${MODEL}.tar.gz

#configs/traffic_config.yaml
#num_classes: 8
#label_map: {1: no_entry, 2: no_stop_park, 3: no_turn, 4: speed_limit, 5: no_remain, 6: warning, 7: signs}


cd ../dataset


python create_coco_tfrecord.py \
  --image_dir=${DATA_PATH} \
  --object_annotations_file=${DATA_ANNOTATIONS_PATH} \
  --image_extend_dir=${EXTEND_DATA_PATH} \
  --object_extend_annotations_file=${DATA_EXTEND_ANNOTATIONS_PATH} \
  --output_file_prefix=../tfrecord/train \
  --num_shards=32





