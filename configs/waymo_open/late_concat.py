_base_ = [
    '../_base_/models/universenet50_2008_late_concat.py',
    '../_base_/datasets/waymo_open_2d_detection_rgbd_f0_mstrain_640_1280.py',
    '../_base_/schedules/schedule_late_concat.py', '../_base_/waymo_runtime.py',
    '../_base_/attacks/pgd.py'
]
model = dict(bbox_head=dict(num_classes=3), pretrained=None)

data = dict(samples_per_gpu=2)

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

# load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.08/universenet50_2008_fp16_4x4_mstrain_480_960_2x_coco_20200815_epoch_24-81356447.pth'  # noqa
load_from = None  # noqa
