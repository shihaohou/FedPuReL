import os
import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import scipy.io

from torch.utils.data import Dataset
from utils.datasplit import partition_data, partition_data_LT
from Dassl.dassl.data.datasets.base_dataset import Datum

# Utilities for building long-tail
from datasets.long_tail import classify_label as _classify_label_indices
from datasets.long_tail import train_long_tail as _train_long_tail
from datasets.long_tail import flatten_list as _flatten_list


def calculate_class_proportions(traindata_cls_counts):
    client_class_proportions = {}

    for client_id, class_counts in traindata_cls_counts.items():
        total_samples = sum(class_counts.values())
        proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        client_class_proportions[client_id] = proportions

    return client_class_proportions


def load_stanford_cars_LT_data(datadir: str, imb_factor: float, imb_type: str):
    """
    Build Stanford Cars long-tail training set and standard test set from annotation files.
    Long-tail is applied only to training split per class using exponential ratio controlled by `imb_factor`.

    Returns tuple with the same signature used by other *_LT loaders:
    (X_train, y_train, X_test, y_test, train_data, test_data, lab2cname, classnames)
    Note: X_train/X_test are not used downstream in this codebase; set to None.
    """
    # Stanford Cars dataset structure
    train_anno_file = os.path.join(datadir, "devkit", "cars_train_annos.mat")
    test_anno_file = os.path.join(datadir, "devkit", "cars_test_annos_withlabels.mat")
    meta_file = os.path.join(datadir, "devkit", "cars_meta.mat")
    
    train_image_dir = os.path.join(datadir, "cars_train")
    test_image_dir = os.path.join(datadir, "cars_test")
    
    assert os.path.exists(train_anno_file), f"Train annotation file not found: {train_anno_file}"
    assert os.path.exists(test_anno_file), f"Test annotation file not found: {test_anno_file}"
    assert os.path.exists(meta_file), f"Meta file not found: {meta_file}"
    assert os.path.isdir(train_image_dir), f"Train images dir not found: {train_image_dir}"
    assert os.path.isdir(test_image_dir), f"Test images dir not found: {test_image_dir}"

    # Load class names from meta file
    meta_data = scipy.io.loadmat(meta_file)
    class_names = [name[0] for name in meta_data['class_names'][0]]

    def read_annotations(anno_file, image_dir):
        """Read annotation .mat file and extract image paths and labels"""
        anno_data = scipy.io.loadmat(anno_file)
        annotations = anno_data['annotations'][0]
        
        items = []
        for anno in annotations:
            filename = anno[0][0]  # Image filename
            label = int(anno[5][0][0]) - 1  # Class label (convert to 0-based)
            
            # Extract car model name from class_names
            class_name = class_names[label].lower()
            
            impath = os.path.join(image_dir, filename)
            
            item = Datum(impath=impath, label=label, classname=class_name)
            items.append(item)
        return items

    # Read train and test data
    train_items = read_annotations(train_anno_file, train_image_dir)
    test_items = read_annotations(test_anno_file, test_image_dir)

    # Get class information
    num_classes = len(class_names)
    
    # Create classnames mapping
    classnames = [name.lower() for name in class_names]
    lab2cname = {cname: idx for idx, cname in enumerate(classnames)}

    # Build list_label2indices for long-tail selection over training items
    list_label2indices = [[] for _ in range(num_classes)]
    for idx, item in enumerate(train_items):
        list_label2indices[item.label].append(idx)

    # Apply long-tail reduction on training indices
    _, list_label2indices_train_new = _train_long_tail(list_label2indices, num_classes, imb_factor, imb_type)
    lt_train_indices = _flatten_list(list_label2indices_train_new)

    # Filter training items to long-tail selection
    train_items_lt = [train_items[i] for i in lt_train_indices]
    
    # Build y_train from the long-tail filtered data to ensure matching lengths
    y_train = np.array([item.label for item in train_items_lt], dtype=np.int64)
    y_test = np.array([item.label for item in test_items], dtype=np.int64)

    # In this codebase, X_* are typically image arrays for CIFAR; for folder datasets we return None
    X_train = None
    X_test = None

    return (X_train, y_train, X_test, y_test, train_items_lt, test_items, lab2cname, classnames)


class StanfordCars_LT(Dataset):
    dataset_dir = "stanford_cars"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 196  # Stanford Cars has 196 classes

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, y_train = partition_data_LT(
            'stanford_cars_LT', self.dataset_dir, 'noniid-labeldir', cfg.DATASET.USERS,
            imb_factor=cfg.DATASET.IMB_FACTOR, imb_type=cfg.DATASET.IMB_TYPE, beta=cfg.DATASET.BETA,
            logdir="./logs/", seed=getattr(cfg, 'SEED', 1))

        for net_id in range(cfg.DATASET.USERS):
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            for sample in range(len(dataidxs_train)):
                federated_train_x[net_id].append(data_train[dataidxs_train[sample]])
            for sample in range(len(dataidxs_test)):
                federated_test_x[net_id].append(data_test[dataidxs_test[sample]])

        client_proportions = calculate_class_proportions(traindata_cls_counts)

        self.client_proportions = client_proportions
        self.y_train = y_train
        self.train_x = data_train
        self.federated_train_x = federated_train_x
        self.federated_test_x = federated_test_x
        self.data_test = data_test
        self.lab2cname = lab2cname
        self.classnames = classnames