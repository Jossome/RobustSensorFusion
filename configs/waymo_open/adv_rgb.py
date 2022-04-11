_base_ = [
    '../_base_/models/universenet50_2008.py',
    '../_base_/datasets/waymo_open_2d_detection_f0_mstrain_640_1280.py',
    '../_base_/schedules/schedule_rgb.py', '../_base_/waymo_runtime.py',
    '../_base_/attacks/pgd.py'
]
model = dict(bbox_head=dict(num_classes=3), pretrained=None)

data = dict(samples_per_gpu=2)

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

load_from = '/home/research/joss/universenet/work_dirs/rgb/latest.pth'  # noqa
