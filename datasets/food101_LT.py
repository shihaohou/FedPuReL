import os
import random
import numpy as np
from typing import List, Tuple

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


def load_food101_LT_data(datadir: str, imb_factor: float, imb_type: str):
    image_dir = os.path.join(datadir, "images")
    assert os.path.isdir(image_dir), f"Food-101 images dir not found: {image_dir}"

    # List classes by subfolders, ignore hidden
    classnames = sorted([d for d in os.listdir(image_dir)
                         if not d.startswith('.') and os.path.isdir(os.path.join(image_dir, d))])
    num_classes = len(classnames)
    lab2cname = {cname: idx for idx, cname in enumerate(classnames)}

    # Collect items per class
    per_class_items: List[List[dict]] = [[] for _ in range(num_classes)]
    for cname in classnames:
        cidx = lab2cname[cname]
        cdir = os.path.join(image_dir, cname)
        files = [f for f in os.listdir(cdir) if not f.startswith('.')]
        files = [os.path.join(cdir, f) for f in files]
        random.shuffle(files)
        for fpath in files:
            per_class_items[cidx].append(Datum(
                impath=fpath,
                label=cidx,
                domain=0,
                classname=cname
            ))

    # Split train/test per class (keep test relatively balanced)
    p_trn = 0.5
    train_items: List[dict] = []
    test_items: List[dict] = []
    for cidx in range(num_classes):
        items = per_class_items[cidx]
        n_total = len(items)
        n_train = max(1, int(round(n_total * p_trn)))
        # ensure test has samples too
        n_train = min(n_train, n_total - 1) if n_total > 1 else n_train
        train_items.extend(items[:n_train])
        test_items.extend(items[n_train:])

    # Build test label array
    y_test = np.array([it.label for it in test_items], dtype=np.int64)

    # Build list_label2indices for long-tail selection over training items
    list_label2indices = [[] for _ in range(num_classes)]
    for idx, it in enumerate(train_items):
        list_label2indices[it.label].append(idx)

    # Apply long-tail reduction on training indices
    _, list_label2indices_train_new = _train_long_tail(list_label2indices, num_classes, imb_factor, imb_type)
    lt_train_indices = _flatten_list(list_label2indices_train_new)

    # Filter training items to long-tail selection
    train_items_lt = [train_items[i] for i in lt_train_indices]
    
    # Build train label array from long-tail filtered data to match train_items_lt length
    y_train = np.array([it.label for it in train_items_lt], dtype=np.int64)

    # In this codebase, X_* are typically image arrays for CIFAR; for folder datasets we return None
    X_train = None
    X_test = None

    return (X_train, y_train, X_test, y_test, train_items_lt, test_items, lab2cname, classnames)


class Food101_LT(Dataset):
    dataset_dir = "food-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 101

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, y_train = partition_data_LT(
            'food101_LT', self.dataset_dir, 'noniid-labeldir', cfg.DATASET.USERS,
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