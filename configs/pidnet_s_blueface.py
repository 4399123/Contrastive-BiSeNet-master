
cfg = dict(
    model_type='pidnet_s',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0002,
    weight_decay=5e-4,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX2',
    train_im_anns='../../BlueFaceDataX2/train.txt',
    val_im_anns='../../BlueFaceDataX2/val.txt',
    scales=[0.75, 1.25],
    cropsize=[512, 512],
    ims_per_gpu=32,
    eval_ims_per_gpu=1,
    use_fp16=False,
    use_sync_bn=True,
    respth='./res',
)