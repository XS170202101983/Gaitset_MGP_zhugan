conf = {
    "WORK_PATH": "./work",#work文件夹路径
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {#/media/user/28b5139d-4fea-4d63-a39f-2630c2ee375f/user/GaitDateset
        'dataset_path': r"/media/user/28b5139d-4fea-4d63-a39f-2630c2ee375f/user/GaitDateset",
        'resolution': '64',#分辨率 数据集分辨率为64*64
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73, # # LT划分方式 4用于训练（In CASIA-B, data of subject #5 is incomplete.），其余(4)的用于测试  //1
        'pid_shuffle': False,# 是否进行随机的划分数据集，如果为False，那么直接选取1-74为训练集，剩余的为测试集
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),#(8,16), #p=8是八个人，k=16是16个视频，每个视频取30帧图片，每个角度一个视频.avi //1,8
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 5,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
