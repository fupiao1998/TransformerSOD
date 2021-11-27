def get_path_dict(hostname, task):
    path_dict = {}
    if hostname == 'LabPC2':
        if task == 'COD':
            path_dict['image_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/Imgs/'
            path_dict['gt_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/GT/'
            path_dict['test_dataset_root'] = '/home1/datasets/SOD_COD/COD/COD_test/'
        elif task == 'SOD':
            path_dict['image_root'] = '/home1/maoyuxin/datasets/SOD/DUTS/img/'
            path_dict['gt_root'] = '/home1/maoyuxin/datasets/SOD/DUTS/gt/'
            path_dict['test_dataset_root'] = '/home1/maoyuxin/datasets/SOD/SOD_RGB/'
        elif task == 'RGBD-SOD':
            path_dict['image_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/RGB/'
            path_dict['gt_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/GT/'
            path_dict['depth_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/depth/'
            path_dict['test_dataset_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/test/'
        elif task == 'Weak-RGB-SOD':
            path_dict['image_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/img/'
            path_dict['gt_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gt/'
            path_dict['mask_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/mask/'
            path_dict['gray_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gray/'
            path_dict['test_dataset_root'] = '/home1/datasets/SOD_COD/SOD_RGB/'
    elif hostname == 'gpu6':
        if task == 'COD':
            path_dict['image_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/Imgs/'
            path_dict['gt_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/GT/'
            path_dict['test_dataset_root'] = '/home1/datasets/SOD_COD/COD/COD_test/'
        elif task == 'SOD':
            path_dict['image_root'] = '/home/maoyuxin/dataset/SOD_COD/DUTS/img/'
            path_dict['gt_root'] = '/home/maoyuxin/dataset/SOD_COD/DUTS/gt/'
            path_dict['test_dataset_root'] = '/home2/dataset/maoyuxin/SOD_COD/SOD_RGB/'
        elif task == 'RGBD-SOD':
            path_dict['image_root'] = '/home/maoyuxin/dataset/SOD_COD/RGBD_SOD/train/RGB/'
            path_dict['gt_root'] = '/home/maoyuxin/dataset/SOD_COD/RGBD_SOD/train/GT/'
            path_dict['depth_root'] = '/home/maoyuxin/dataset/SOD_COD/RGBD_SOD/train/depth/'
            path_dict['test_dataset_root'] = '/home/maoyuxin/dataset/SOD_COD/RGBD_SOD/test/'
        elif task == 'Weak-RGB-SOD':
            path_dict['image_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/img/'
            path_dict['gt_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gt/'
            path_dict['mask_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/mask/'
            path_dict['gray_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gray/'
            path_dict['test_dataset_root'] = '/home1/datasets/SOD_COD/SOD_RGB/'
    
    return path_dict
