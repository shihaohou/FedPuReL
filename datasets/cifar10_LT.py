import os

from torch.utils.data import Dataset
from utils.datasplit import partition_data, partition_data_LT


def calculate_class_proportions(traindata_cls_counts):
    client_class_proportions = {}

    for client_id, class_counts in traindata_cls_counts.items():
        total_samples = sum(class_counts.values())
        proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        client_class_proportions[client_id] = proportions

    return client_class_proportions

class Cifar10_LT(Dataset):
    dataset_dir = "cifar-10"
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10

        federated_train_x = [[] for i in range(cfg.DATASET.USERS)]
        federated_test_x = [[] for i in range(cfg.DATASET.USERS)]

        data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, y_train = partition_data_LT(
            'cifar10_LT', self.dataset_dir, cfg.DATASET.PARTITION, cfg.DATASET.USERS, beta=cfg.DATASET.BETA, logdir="./logs/", imb_factor=cfg.DATASET.IMB_FACTOR, imb_type=cfg.DATASET.IMB_TYPE, seed=getattr(cfg, 'SEED', 1))

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


