# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[64, 73, 83, 200])
total_epochs = 200
