import os
import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict

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


def load_oxford_pets_LT_data(datadir: str, imb_factor: float, imb_type: str):
    """
    Build Oxford Pets long-tail training set and standard test set from annotation files.
    Long-tail is applied only to training split per class using exponential ratio controlled by `imb_factor`.

    Returns tuple with the same signature used by other *_LT loaders:
    (X_train, y_train, X_test, y_test, train_data, test_data, lab2cname, classnames)
    Note: X_train/X_test are not used downstream in this codebase; set to None.
    """
    image_dir = os.path.join(datadir, "images")
    anno_dir = os.path.join(datadir, "annotations")
    
    assert os.path.isdir(image_dir), f"Oxford Pets images dir not found: {image_dir}"
    assert os.path.isdir(anno_dir), f"Oxford Pets annotations dir not found: {anno_dir}"

    def read_data(split_file):
        filepath = os.path.join(anno_dir, split_file)
        items = []
        
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                
                items.append(Datum(
                    impath=impath,
                    label=label,
                    domain=0,
                    classname=breed
                ))
        return items

    # Read trainval and test data
    trainval_items = read_data("trainval.txt")
    test_items = read_data("test.txt")

    # Split trainval into train and val (use train for long-tail processing)
    p_val = 0.2
    p_trn = 1 - p_val
    
    # Group by label for splitting
    tracker = defaultdict(list)
    for idx, item in enumerate(trainval_items):
        label = item.label
        tracker[label].append(idx)

    train_items = []
    val_items = []
    
    for label, idxs in tracker.items():
        n_val = round(len(idxs) * p_val)
        n_val = max(1, n_val)  # ensure at least 1 validation sample
        random.shuffle(idxs)
        
        for n, idx in enumerate(idxs):
            item = trainval_items[idx]
            if n < n_val:
                val_items.append(item)
            else:
                train_items.append(item)

    # Get class information
    all_labels = set()
    for item in trainval_items + test_items:
        all_labels.add(item.label)
    
    num_classes = len(all_labels)
    
    # Create classnames mapping
    label_to_classname = {}
    for item in trainval_items + test_items:
        label_to_classname[item.label] = item.classname
    
    classnames = [label_to_classname[i] for i in range(num_classes)]
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
    
    # CRITICAL FIX: Build y_train from the long-tail filtered data, not original data
    # This ensures y_train and data_train have matching lengths
    y_train = np.array([item.label for item in train_items_lt], dtype=np.int64)
    y_test = np.array([item.label for item in test_items], dtype=np.int64)

    # In this codebase, X_* are typically image arrays for CIFAR; for folder datasets we return None
    X_train = None
    X_test = None

    return (X_train, y_train, X_test, y_test, train_items_lt, test_items, lab2cname, classnames)


class OxfordPets_LT(Dataset):
    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 37  # Oxford Pets has 37 classes

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, y_train = partition_data_LT(
            'oxford_pets_LT', self.dataset_dir, 'noniid-labeldir', cfg.DATASET.USERS,
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