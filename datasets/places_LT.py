import os
import pickle
import numpy as np
from collections import OrderedDict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import listdir_nohidden, mkdir_if_missing
from collections import Counter


def calculate_class_proportions(traindata_cls_counts):
    client_class_proportions = {}

    for client_id, class_counts in traindata_cls_counts.items():
        total_samples = sum(class_counts.values())
        proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        client_class_proportions[client_id] = proportions

    return client_class_proportions

def get_class_distribution(train_data):
    # 使用列表推导式获取所有的标签
    labels = [item.label for item in train_data]

    # 使用 Counter 来计算每个标签的出现次数
    class_distribution = Counter(labels)

    # 将结果转换为有序字典，按标签排序
    sorted_distribution = dict(sorted(class_distribution.items()))

    return sorted_distribution

# @DATASET_REGISTRY.register()
class Places_LT(DatasetBase):
    loader_dir = ""
    dataset_dir = ""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.placesROOT))
        root = "/root/autodl-tmp/FLRFT/DATA/places"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.loader_dir = os.path.join(root, self.loader_dir)
        
        # 读取类名
        classnames_file = os.path.join(self.loader_dir, "class_name_list.txt")
        classnames = self.read_classnames(classnames_file)
        
        # 读取训练和测试数据
        train = self.read_data(os.path.join(self.loader_dir, "Places_LT_train.txt"), classnames)
        test = self.read_data(os.path.join(self.loader_dir, "Places_LT_test.txt"), classnames)
        
        # 获取类别分布
        class_distribution = get_class_distribution(train)
        print("class_distribution:", class_distribution)
        
        # 生成联邦数据集
        federated_train_x, traindata_cls_counts = self.generate_federated_dataset_imagenet(
            train, 
            num_users=cfg.DATASET.USERS, 
            is_iid=cfg.DATASET.IID, 
            beta=cfg.DATASET.BETA
        )
        
        # 为测试数据创建联邦化分割
        federated_test_x, testdata_cls_counts = self.generate_federated_dataset_imagenet(
            test, 
            num_users=cfg.DATASET.USERS, 
            is_iid=cfg.DATASET.IID, 
            beta=cfg.DATASET.BETA
        )
        
        client_proportions = calculate_class_proportions(traindata_cls_counts)
        
        self.y_train = np.array([item.label for item in train])
        self.data_test = test
        self.client_proportions = client_proportions
        
        super().__init__(train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x, test=test)

    @staticmethod
    def read_classnames(text_file):
        """读取Places_LT的类名文件"""
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ", 1)  # 只分割第一个空格
                label = int(line[0])
                classname = line[1]
                classnames[label] = classname
        return classnames

    def read_data(self, split_file, classnames):
        """读取训练或测试数据"""
        items = []
        
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    impath, label = parts
                    label = int(label)
                    
                    # 获取完整路径
                    full_impath = os.path.join(self.dataset_dir, impath)
                    
                    # 获取类名
                    classname = classnames.get(label, f"class_{label}")
                    
                    item = Datum(impath=full_impath, label=label, classname=classname)
                    items.append(item)
                else:
                    print(f"Warning: Skipping invalid line: {line.strip()}")
        
        return items