# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import mmcv
import json


def _create_pd_file_example(preds):
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()

    for pred in preds['annotations']:
        """
        a['annotations'][0]: {'id': 0, 'image_id': 0, 'category_id': 1, 'segmentation': None, 'area': 969.341405, 'bbox': [536.535705, 623.37933, 51.15879, 18.9477], 'iscrowd': 0, 'track_id': '037a0ea9-5597-4a75-9375-4fc0dc1e7c94', 'det_difficult': 2, 'track_difficult': 2}

        a['images'][0]: {'id': 0, 'width': 1920, 'height': 1280, 'file_name': '00000_00000_camera1.jpg', 'license': 1, 'flickr_url': '', 'coco_url': '', 'date_captured': '', 'context_name': '10203656353524179475_7625_000_7645_000', 'timestamp_micros': 1522688014970187, 'camera_id': 1, 'sequence_id': 0, 'frame_id': 0, 'time_of_day': 'Day', 'location': 'location_phx', 'weather': 'sunny'}
        """
        o = metrics_pb2.Object()
        img = preds['images'][pred['image_id']]
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        o.context_name = img['context_name']
        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        invalid_ts = -1
        o.frame_timestamp_micros = img['timestamp_micros']
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
        o.camera_name = img['camera_id']

        # Populating box and score.
        box = label_pb2.Label.Box()
        bb = pred['bbox']
        box.center_x = bb[0] + bb[2] / 2
        box.center_y = bb[1] + bb[3] / 2
        box.length = bb[2]
        box.width = bb[3]
        o.object.box.CopyFrom(box)
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        # For tracking, this must be set and it must be unique for each tracked
        # sequence.
        o.object.id = pred['track_id']
        # Use correct type.
        if pred['category_id'] == 3:
            o.object.type = 4
        else:
            o.object.type = pred['category_id']
        o.object.num_lidar_points_in_box = 100
        o.object.detection_difficulty_level = pred['det_difficult']

        objects.objects.append(o)

        # Add more objects. Note that a reasonable detector should limit its maximum
        # number of boxes predicted per frame. A reasonable value is around 400. A
        # huge number of boxes can slow down metrics computation.

    # Write objects to a file.
    f = open('gt2d.bin', 'wb')
    f.write(objects.SerializeToString())
    f.close()


def main():
    gt = mmcv.load('/home/research/joss/Kitti/waymo/coco/annotations/subset200_instances_val2020.json')
    _create_pd_file_example(gt)


if __name__ == '__main__':
    main()
