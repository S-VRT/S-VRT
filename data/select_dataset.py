

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
    if dataset_type in ['videorecurrenttraindataset']:
        from data.dataset_video_train import VideoRecurrentTrainDataset as D
    elif dataset_type in ['videorecurrenttrainnonblinddenoisingdataset']:
        from data.dataset_video_train import VideoRecurrentTrainNonblindDenoisingDataset as D
    elif dataset_type in ['videorecurrenttrainvimeodataset']:
        from data.dataset_video_train import VideoRecurrentTrainVimeoDataset as D
    elif dataset_type in ['videorecurrenttrainvimeovfidataset']:
        from data.dataset_video_train import VideoRecurrentTrainVimeoVFIDataset as D
    elif dataset_type in ['videorecurrenttraindatasetrgbspike']:
        from data.dataset_video_train_rgbspike import VideoRecurrentTrainDatasetRGBSpike as D
    elif dataset_type in ['videorecurrenttestdataset']:
        from data.dataset_video_test import VideoRecurrentTestDataset as D
    elif dataset_type in ['singlevideorecurrenttestdataset']:
        from data.dataset_video_test import SingleVideoRecurrentTestDataset as D
    elif dataset_type in ['videotestvimeo90kdataset']:
        from data.dataset_video_test import VideoTestVimeo90KDataset as D
    elif dataset_type in ['vfi_davis']:
        from data.dataset_video_test import VFI_DAVIS as D
    elif dataset_type in ['vfi_ucf101']:
        from data.dataset_video_test import VFI_UCF101 as D
    elif dataset_type in ['vfi_vid4']:
        from data.dataset_video_test import VFI_Vid4 as D

    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
