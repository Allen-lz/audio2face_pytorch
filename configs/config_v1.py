config = {
        'gpu_ids': "0",  # 使用的GPU序号
        'lr': 0.00001,  # 0.0005, 0.01
        'class_num': 16,
        'ckpt': "experiment/checkpoints/best_model_loss_2.51.pth",
        'lr_update_interval': 30000,  # 学习率更新频率 每次*0.99
        'warmup_steps': 0,  # 前200个iteration使用warmup, 之后使用正常的学习率
        'watch_interval': 1000,  # log打印
        'print_loss': 1000,  # loss打印的频率
        'val_interval': 100000,  # 验证轮次
        'save_interval': 100000,  # 保存模型轮次
        'epoch': 10000,  #
        'exp_dir': 'experiment/exp_1',
        'train_batch_size': 64,  # 64
        'num_workers': 8,
        'train_target_root': "E:/datasets/audio2face/train_gt",
        'train_data_root': "E:/datasets/audio2face/train_data",
        'val_target_root': "E:/datasets/audio2face/val_gt",
        'val_data_root': "E:/datasets/audio2face/val_data",
}