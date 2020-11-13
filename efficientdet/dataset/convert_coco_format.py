import json
import cv2
import os
import pickle as pkl

data_dir = '/content/drive/My Drive/TSD/Yet-Another-EfficientDet-Pytorch/datasets/sample'
traffic_dir = "/content/drive/My Drive/TSD/Yet-Another-EfficientDet-Pytorch/datasets/traffic"
traffic_data = json.load(open(traffic_dir + '/annotations/instances_train2017.json'))
img_ids = [img['id'] for img in traffic_data['images']]
ann_ids = [ann['id'] for ann in traffic_data['annotations']]
current_max_image_id = max(img_ids)
current_max_ann_id = max(ann_ids)
original_size = (1628, 1236)
target_size = (1622, 626)
keep_size = (8, 1236 - 626)
mapping_idx = {}
orignal_label_map = {1: "no_entry", 2: "no_stop_park", 3: "no_turn",
                     4: "speed_limit", 5: "no_remain", 6: "warning", 7: "signs"}

extend_label_map = {1: "speed_limit", 2: "goods_vehicles", 3: "no_overtaking",
                    4: "no_stopping", 5: "no_parking", 6: "stop", 7: "bicycle", 8: "hump", 9: "no_left", 10: "no_right",
                    11: "priority_to", 12: "no_entry", 13: "yield", 14: "parking"}

label_map_mapping = {1: 4, 4: 2, 5: 2, 9: 3, 10: 3, 12: 1, 3: 5, 14: 6, 7: 7, 11: 6, 8: 7, 2: 6, 6: 6, 13: 6}

anns = json.load(open(data_dir + '/sample.json', 'r'))
convert_annotations = {}
convert_annotations['info'] = {}
convert_annotations['images'] = []
convert_annotations['annotations'] = []
convert_annotations['categories'] = []
for idx, ann in enumerate(anns):
    file_name = ann['filename']
    img = cv2.imread(os.path.join(data_dir, 'images', file_name))
    crop_img = img[0:target_size[1], 0:target_size[0]]
    width = target_size[0]
    height = target_size[1]
    _ann = ann['ann']
    new_image_id = idx + current_max_image_id + 1
    mapping_idx[new_image_id] = file_name
    cv2.imwrite('../data/cure/images/{}.png'.format(str(new_image_id)), crop_img)

    convert_annotations['images'].append({'file_name': str(new_image_id) + '.png',
                                          'height': height,
                                          'width': width,
                                          'id': new_image_id})
    for i in range(len(_ann['labels'])):
        if _ann['bboxes'][i][1] > target_size[1] or _ann['bboxes'][i][0] > target_size[0]:
            continue

        current_max_ann_id += 1
        _ann['bboxes'][i][2] = _ann['bboxes'][i][2] if _ann['bboxes'][i][2] <= target_size[0] else target_size[0]
        _ann['bboxes'][i][3] = _ann['bboxes'][i][3] if _ann['bboxes'][i][3] <= target_size[1] else target_size[1]

        convert_annotations['annotations'].append({'image_id': new_image_id,
                                                   'category_id': label_map_mapping[int(_ann['labels'][i])],
                                                   'id': current_max_ann_id,
                                                   'bbox': [_ann['bboxes'][i][0], _ann['bboxes'][i][1],
                                                            _ann['bboxes'][i][2] - _ann['bboxes'][i][0],
                                                            _ann['bboxes'][i][3] - _ann['bboxes'][i][1]]})


json.dump(convert_annotations, open('../data/cure/annotations/cure_annotations.json', 'w+'))
pkl.dump(mapping_idx, open('../data/cure/mapping_idx.pkl', 'wb'))
