

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    phase = str(dataset_opt.get('phase', '')).lower()
    
    # -----------------------------------------
    # video restoration
    # -----------------------------------------
    if dataset_type in ['traindataset']:
        from data.dataset_video_train import TrainDataset as D
    elif dataset_type in ['traindatasetrgbspike']:
        if phase == 'test':
            from data.dataset_video_test import TrainDatasetRGBSpike as D
        else:
            from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike as D
    elif dataset_type in ['testdataset']:
        from data.dataset_video_test import TestDataset as D
    elif dataset_type in ['singletestdataset']:
        from data.dataset_video_test import SingleTestDataset as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
