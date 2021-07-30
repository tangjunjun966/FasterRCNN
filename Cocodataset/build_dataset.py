from Cocodataset.cocodata import CocoDataSet, DataTest
from torch.utils.data import DataLoader

import torch


class Build_Dataset():
    def __init__(self):
        self.mode = None

    def build_train_data(self, data_root, train_pipelines, mode='train'):
        # data_root文件夹中必须包含cocojson文件，如训练为 train.json,暂时不开接口
        datasets_train = CocoDataSet(data_root, train_pipelines, mode=mode)
        self.mode = mode
        return datasets_train

    def build_val_data(self, data_root, val_pipelines, mode='val'):
        datasets_val = CocoDataSet(data_root, val_pipelines, mode=mode)
        self.mode = mode
        return datasets_val

    def build_test_data(self, data_root, test_pipelines, mode='test'):
        datasets_test = CocoDataSet(data_root, test_pipelines, mode=mode)
        self.mode = mode
        return datasets_test

    def build_img_data(self, test_pipelines, mode='test'):
        # 构建一张图测试的类初始化
        data_img_test = DataTest(test_pipelines, mode=mode)
        self.mode = mode
        return data_img_test

    def collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        num_batch = len(batch)
        img_info_keys = batch[0]['img_info']
        img_meta_keys = batch[0]['img_meta']
        data = {}
        for key in img_info_keys:
            key_lst = []
            for i in range(num_batch):
                key_lst.append(batch[i][key])
            if key == 'img':
                key_lst = torch.stack(key_lst, 0)
            data[key] = key_lst
        img_meta = []
        for i in range(num_batch):
            key_dict = {}
            for key in img_meta_keys:
                key_dict[key] = batch[i][key]
            img_meta.append(key_dict)

        return [data, img_meta]

    def build_dataloader(self, datasets, **kwargs):

        batch_size = kwargs.get('batch_size', 1)
        num_worksers = kwargs.get('num_workers', 2)
        shuffle = kwargs.get('shffle', True)
        drop_last = kwargs.get('drop_last', True)

        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_worksers,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                collate_fn=self.collate
                                )

        return dataloader



if __name__ == '__main__':
    test_pipeline = [
        dict(type='Resize', img_scale=(256, 260)),
        dict(type='Normalize'),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])

    ]

    train_pipeline = [
        dict(type='LoadImageFromFile'),  # 载入图像
        dict(type='LoadAnnotations', with_bbox=True),  # 载入annotations

        dict(type='Resize', img_scale=(512, 512)),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='RandomFlip', direction='vertical', flip_ratio=0.5),

        dict(type='RandomCropResize', crop_size=(426, 426), crop_ratio=1.1),

        # 加载数据处理模块#

        dict(type='Normalize'),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
             img_meta_keys=['cat2class']),  # 在results中需要提取的结果
    ]

    dataset_json = False
    if dataset_json:
        # Cocojson数据处理过程
        root = r'C:\Users\51102\Desktop\Fasterrcnn_tj\Cocodataset\train_data'
        BD = Build_Dataset()
        dataset = BD.build_train_data(root, train_pipeline)
        dataloader = BD.build_dataloader(dataset)
        for data, img_meta in dataloader:
            print(data['img'].shape)
            print(img_meta)
    else:
        img_root = r'C:\Users\51102\Desktop\Fasterrcnn_tj\Cocodataset\train_data\A1WOP\L3MA12E19011AA0_11620_2_787.206_656.623_PARTICLE_20210215_171935_REV_A1WOP_RS.JPG'
        BD = Build_Dataset()
        dataset_img = BD.build_img_data(test_pipeline)
        result_img = dataset_img.data_test(img_root)
        print(result_img)
