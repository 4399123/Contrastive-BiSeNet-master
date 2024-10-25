
cfg = dict(
    model_type='topformer_hgnetv2_b5',   #使用 one 训练
    n_cats=9,
    num_aux_heads=1,
    lr_start=0.0001,
    weight_decay=5e-4,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX2',
    train_im_anns='../../BlueFaceDataX2/train.txt',
    val_im_anns='../../BlueFaceDataX2/val.txt',
    scales=[0.85, 1.25],
    cropsize=[512, 512],
    ims_per_gpu=48,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
