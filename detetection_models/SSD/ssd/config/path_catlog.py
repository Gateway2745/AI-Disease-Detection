import os


class DatasetCatalog:
    DATA_DIR = 'datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'my_dataset_train': {
            "data_dir": "train-set-5",
            "ann_file": "annotations/train-set-5.json",
        },
        'my_dataset_val': {
            "data_dir": "val-set-1",
            "ann_file": "annotations/val-set-1.json",
        },
        'my_test_1': {
            "data_dir": "test-set-1/images",
            "ann_file": "annotations/test-set-1.json",
        },
        'my_test_2': {
            "data_dir": "test-set-2/images",
            "ann_file": "annotations/test-set-2.json",
        },
        'my_test_3': {
            "data_dir": "test-set-3/images",
            "ann_file": "annotations/test-set-3.json",
        },
        'my_test_4': {
            "data_dir": "test-set-4/images",
            "ann_file": "annotations/test-set-4.json",
        },
        'my_test_5': {
            "data_dir": "test-set-5/images",
            "ann_file": "annotations/test-set-5.json",
        }
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)
        elif 'my' in name:
            root_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root_dir, attrs["data_dir"]),
                ann_file=os.path.join(root_dir, attrs["ann_file"]),
            )
            return dict(factory="MyDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
