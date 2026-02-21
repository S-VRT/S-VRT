

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    
    # -----------------------------------------
    # video restoration
    # -----------------------------------------
    if dataset_type in ['TrainDataset']:
        from data.dataset_video_train import TrainDataset as D
    elif dataset_type in ['TrainDatasetRGBSpike']:
        from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike as D
    elif dataset_type in ['TestDataset']:
        from data.dataset_video_test import TestDataset as D
    elif dataset_type in ['singleTestDataset']:
        from data.dataset_video_test import SingleTestDataset as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
