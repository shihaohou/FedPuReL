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


def load_aircraft_LT_data(datadir: str, imb_factor: float, imb_type: str):
    """
    Build FGVC-Aircraft long-tail training set and standard test set from annotation files.
    Long-tail is applied only to training split per class using exponential ratio controlled by `imb_factor`.

    Returns tuple with the same signature used by other *_LT loaders:
    (X_train, y_train, X_test, y_test, train_data, test_data, lab2cname, classnames)
    Note: X_train/X_test are not used downstream in this codebase; set to None.
    """
    # Resolve image directory (supports official 'data/images' and root 'images')
    image_dir_candidates = [
        os.path.join(datadir, "images"),
        os.path.join(datadir, "data", "images"),
    ]
    image_dir = next((p for p in image_dir_candidates if os.path.isdir(p)), None)
    if image_dir is None:
        raise AssertionError(f"FGVC-Aircraft images dir not found under: {image_dir_candidates}")

    # Resolve annotation files
    trainval_root_candidates = [
        os.path.join(datadir, "images_variant_trainval.txt"),
        os.path.join(datadir, "data", "images_variant_trainval.txt"),
    ]
    test_root_candidates = [
        os.path.join(datadir, "images_variant_test.txt"),
        os.path.join(datadir, "data", "images_variant_test.txt"),
    ]
    # Fallback to separate train/val files if trainval not found
    train_files_sep_candidates = [
        (os.path.join(datadir, "data", "images_variant_train.txt"),
         os.path.join(datadir, "data", "images_variant_val.txt")),
        (os.path.join(datadir, "images_variant_train.txt"),
         os.path.join(datadir, "images_variant_val.txt")),
    ]

    def pick_existing_file(paths):
        return next((p for p in paths if os.path.exists(p)), None)

    trainval_file = pick_existing_file(trainval_root_candidates)
    test_file = pick_existing_file(test_root_candidates)

    # Build list of training annotation files
    if trainval_file:
        train_ann_files = [trainval_file]
    else:
        # pick first pair of existing train/val files
        train_ann_files = None
        for tr, va in train_files_sep_candidates:
            if os.path.exists(tr) and os.path.exists(va):
                train_ann_files = [tr, va]
                break
        if train_ann_files is None:
            raise AssertionError(
                "Train list files not found. Checked: "
                + ", ".join(trainval_root_candidates + [p for pair in train_files_sep_candidates for p in pair])
            )
    if not test_file:
        raise AssertionError("Test list file not found. Checked: " + ", ".join(test_root_candidates))

    def read_annotation_files(ann_files):
        """Read one or multiple annotation files and extract image paths and labels"""
        def resolve_image_path(image_dir: str, image_name: str) -> str:
            # Normalize and try common patterns: raw name, add extension, strip prefix
            raw = image_name.strip().lstrip('./')
            base = os.path.basename(raw)
            candidates = [
                os.path.join(image_dir, raw),
                os.path.join(image_dir, base),
                os.path.join(image_dir, base + ".jpg"),
                os.path.join(image_dir, base + ".jpeg"),
                os.path.join(image_dir, base + ".png"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    return p
            # Default to .jpg under images dir if nothing matched
            return os.path.join(image_dir, base + ".jpg")

        items = []
        classnames_set = set()
        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                image_name, variant_name = parts
                classnames_set.add(variant_name)
                impath = resolve_image_path(image_dir, image_name)
                items.append({'impath': impath, 'variant_name': variant_name})
        return items, sorted(list(classnames_set))

    # Read train and test data
    train_items_raw, train_classnames = read_annotation_files(train_ann_files)
    test_items_raw, test_classnames = read_annotation_files([test_file])
    
    # Combine and sort all unique classnames
    all_classnames = sorted(list(set(train_classnames + test_classnames)))
    num_classes = len(all_classnames)
    
    # Create label mapping
    classname_to_label = {name: idx for idx, name in enumerate(all_classnames)}
    lab2cname = {name: idx for idx, name in enumerate(all_classnames)}
    classnames = all_classnames

    # Convert to Datum objects with proper labels
    def convert_to_datum(items_raw):
        items = []
        for item in items_raw:
            label = classname_to_label[item['variant_name']]
            datum = Datum(
                impath=item['impath'],
                label=label,
                domain=0,
                classname=item['variant_name']
            )
            items.append(datum)
        return items

    train_items = convert_to_datum(train_items_raw)
    test_items = convert_to_datum(test_items_raw)

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


class Aircraft_LT(Dataset):
    dataset_dir = "fgvc-aircraft-2013b"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 100  # FGVC-Aircraft has 100 variant classes

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, y_train = partition_data_LT(
            'aircraft_LT', self.dataset_dir, 'noniid-labeldir', cfg.DATASET.USERS,
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