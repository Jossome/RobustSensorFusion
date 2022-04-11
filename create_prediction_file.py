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


def _create_pd_file_example(preds):
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()

    for pred in preds:
        """
        {'context_name': '10203656353524179475_7625_000_7645_000', 'timestamp_micros': 1522688014970187, 'camera_name': 1, 'frame_index': 0, 'center_x': 245.4140625, 'center_y': 684.7306518554688, 'length': 108.099609375, 'width': 67.0257568359375, 'score': 0.97314453125, 'type': 1, 'id': '0_0_0'}
        """
        o = metrics_pb2.Object()
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        o.context_name = pred['context_name']
        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        invalid_ts = -1
        o.frame_timestamp_micros = pred['timestamp_micros']
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
        o.camera_name = pred['camera_name']

        # Populating box and score.
        box = label_pb2.Label.Box()
        box.center_x = pred['center_x']
        box.center_y = pred['center_y']
        box.length = pred['length']
        box.width = pred['width']
        o.object.box.CopyFrom(box)
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        o.score = pred['score']
        # For tracking, this must be set and it must be unique for each tracked
        # sequence.
        o.object.id = pred['id']
        # Use correct type.
        o.object.type = pred['type']

        o.object.num_lidar_points_in_box = 100
        objects.objects.append(o)

        # Add more objects. Note that a reasonable detector should limit its maximum
        # number of boxes predicted per frame. A reasonable value is around 400. A
        # huge number of boxes can slow down metrics computation.

    # Write objects to a file.
    f = open('submission_preds.bin', 'wb')
    f.write(objects.SerializeToString())
    f.close()


def main():
    preds = mmcv.load('waymo_pred.bbox.pkl')
    _create_pd_file_example(preds)


if __name__ == '__main__':
    main()
