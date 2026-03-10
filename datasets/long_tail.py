import numpy as np

import copy
import random
import torch

def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res
def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
def classify_label(dataset, num_classes: int):#hshs
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))

    print("\nNumber of images per class:")
    for class_idx, img_count in enumerate(img_num_list):
        print(f"Class {class_idx}: {img_count} images")

    return img_num_list, list_clients_indices


def train_long_tail_fmnist(list_label2indices_train, num_classes, imb_factor, imb_type):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    list_clients_indices = []
    classes = [8, 1, 7, 5, 9, 3, 0, 2, 4, 6]
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    print("\nNumber of images per class:")
    for class_idx, img_count in enumerate(img_num_list):
        print(f"Class {class_idx}: {img_count} images")
    return img_num_list, list_clients_indices





