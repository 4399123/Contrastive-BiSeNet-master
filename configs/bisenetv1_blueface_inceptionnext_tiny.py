
cfg = dict(
    model_type='bisenetv1_inceptionnext_tiny',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0002,
    weight_decay=4e-4,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX',
    train_im_anns='../../BlueFaceDataX/train.txt',
    val_im_anns='../../BlueFaceDataX/val.txt',
    scales=[0.85, 1.25],
    cropsize=[512, 512],
    ims_per_gpu=40,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
