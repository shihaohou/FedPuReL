import os
import pickle
import numpy as np
from collections import OrderedDict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import listdir_nohidden, mkdir_if_missing
from collections import Counter, defaultdict



def calculate_class_proportions(traindata_cls_counts):
    client_class_proportions = {}

    for client_id, class_counts in traindata_cls_counts.items():
        total_samples = sum(class_counts.values())
        proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        client_class_proportions[client_id] = proportions

    return client_class_proportions
def normalize_path(path):
    """Normalize path to use forward slashes."""
    return path.replace('\\', '/')

def get_class_distribution(train_data):
    # 使用列表推导式获取所有的标签
    labels = [item.label for item in train_data]

    # 使用 Counter 来计算每个标签的出现次数
    class_distribution = Counter(labels)

    # 将结果转换为有序字典，按标签排序
    sorted_distribution = dict(sorted(class_distribution.items()))

    return sorted_distribution

# @DATASET_REGISTRY.register()
class ImageNet_LT(DatasetBase):
    loader_dir = "ImageNet_LT"
    # loader_dir = "imagenet"
    dataset_dir = ""

    def __init__(self, cfg):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.imagenetROOT))
        root = "/root/autodl-tmp/imagenet"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.loader_dir = os.path.join(root, self.loader_dir)
        self.image_dir_train = os.path.join(self.dataset_dir, 'train')
        self.image_dir_test = os.path.join(self.dataset_dir)
        # self.class_distribution = self.get_class_distribution()
        #
        # print("class_distribution",self.class_distribution)

        text_file = os.path.join(self.loader_dir, "classnames.txt")


        classnames = self.read_classnames(text_file)
        train = self.read_train_data(classnames, "ImageNet_LT_train.txt")
        test, label_to_classname = self.read_test_data(classnames, "ImageNet_LT_test.txt")

        class_distribution = get_class_distribution(train)
        print("class_distribution:",class_distribution)
        # for label, count in class_distribution.items():
        #     print(f"Class {label}: {count} samples")
        # classname2label = {name: label for label, name in label_to_classname.items()}
        # classnames = [name for _, name in sorted([(label, name) for name, label in classname2label.items()])]

        federated_train_x, traindata_cls_counts = self.generate_federated_dataset_imagenet(train, num_users=cfg.DATASET.USERS, is_iid=cfg.DATASET.IID, beta=cfg.DATASET.BETA)
        
        # 为测试数据创建平衡的联邦化分割
        federated_test_x, testdata_cls_counts = self.generate_balanced_federated_test(test, num_users=cfg.DATASET.USERS)
        
        client_proportions = calculate_class_proportions(traindata_cls_counts)
        # self.federated_train_x = federated_train_x
        # self.lab2name = classname2label
        # self.classnames = classnames
        # self.train_x = train
        self.y_train = np.array([item.label for item in train])
        self.data_test = test
        self.client_proportions = client_proportions
        super().__init__(train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x, test=test)

    @staticmethod
    def read_classnames(text_file):
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_train_data(self, classnames, split_file):
        split_file = os.path.join(self.loader_dir, split_file)
        items = []

        # Read the split file to get the mapping of image paths to labels
        image_to_label = {}
        with open(split_file, "r") as f:
            for line in f:
                impath, label = line.strip().split()
                image_to_label[impath] = int(label)

        # Get all folders (classes) in the image directory
        folders = sorted(f.name for f in os.scandir(self.image_dir_train) if f.is_dir())


        for folder in folders:
            classname = classnames[folder]
            imnames = listdir_nohidden(os.path.join(self.image_dir_train, folder))

            for imname in imnames:
                relative_impath = normalize_path(os.path.join('train', folder, imname))
                full_impath = os.path.join(self.image_dir_train, folder, imname)

                if relative_impath in image_to_label:
                    label = image_to_label[relative_impath]
                    item = Datum(impath=full_impath, label=label, classname=classname)
                    items.append(item)

        return items

    # def read_test_data(self, classnames, split_file):
    #     split_file = os.path.join(self.loader_dir, split_file)
    #     items = []
    #
    #     # Create a mapping from label numbers to class names
    #     label_to_classname = {i: name for i, name in enumerate(classnames.values())}
    #
    #     with open(split_file, "r") as f:
    #         for line in f:
    #             file_path, label = line.strip().split()
    #             label = int(label)
    #
    #             # Adjust the file path to match the actual structure
    #             # Remove the 'nXXXXXXXX/' part from the path
    #             adjusted_path = '/'.join(file_path.split('/')[0::2])
    #
    #             # Get the full path to the image
    #             full_impath = os.path.join(self.image_dir_test, adjusted_path)
    #
    #             # Get the classname using the label
    #             classname = label_to_classname[label]
    #
    #             # Create a Datum object only if the file exists
    #             if os.path.exists(full_impath):
    #                 item = Datum(impath=full_impath, label=label, classname=classname)
    #                 items.append(item)
    #             else:
    #                 print(f"Warning: File not found: {full_impath}")
    #
    #     return items, label_to_classname

    def read_test_data(self, classnames, split_file):
        split_file = os.path.join(self.loader_dir, split_file)
        items = []

        # Create a mapping from label numbers to class names
        label_to_classname = {i: name for i, name in enumerate(classnames.values())}

        with open(split_file, "r") as f:
            for line in f:
                file_path, label = line.strip().split()
                label = int(label)

                # Get the full path to the image
                # No need to adjust the path, use it directly as provided in the split file
                full_impath = os.path.join(self.image_dir_test, file_path)

                # Get the classname using the label
                classname = label_to_classname[label]

                # Create a Datum object only if the file exists
                if os.path.exists(full_impath):
                    item = Datum(impath=full_impath, label=label, classname=classname)
                    items.append(item)
                else:
                    print(f"Warning: File not found: {full_impath}")

        return items, label_to_classname
    
    def generate_balanced_federated_test(self, test_data, num_users):
        """
        将平衡的测试集平均分配给各个客户端
        确保每个客户端的测试集包含所有类别且不重复
        
        Args:
            test_data: 测试数据列表
            num_users: 客户端数量
        
        Returns:
            federated_test_x: 每个客户端的测试数据列表
            testdata_cls_counts: 每个客户端的类别统计
        """
        print(f"Creating balanced federated test dataset for {num_users} users")
        
        # 按类别组织测试数据
        class_data = defaultdict(list)
        for item in test_data:
            class_data[item.label].append(item)
        
        num_classes = len(class_data)
        print(f"Number of classes: {num_classes}")
        
        # 检查测试集是否平衡
        class_counts = {label: len(items) for label, items in class_data.items()}
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        print(f"Class counts range: {min_count}-{max_count}")
        
        # 初始化输出
        federated_test_x = [[] for _ in range(num_users)]
        testdata_cls_counts = {i: defaultdict(int) for i in range(num_users)}
        
        # 为每个类别分配样本到客户端
        for label, items in class_data.items():
            # 计算每个客户端应该分配的样本数
            samples_per_user = len(items) // num_users
            remaining_samples = len(items) % num_users
            
            # 打乱该类别的样本顺序
            shuffled_items = items.copy()
            np.random.shuffle(shuffled_items)
            
            # 分配样本
            start_idx = 0
            for user_id in range(num_users):
                # 前 remaining_samples 个客户端多分配一个样本
                if user_id < remaining_samples:
                    n_samples = samples_per_user + 1
                else:
                    n_samples = samples_per_user
                
                # 获取该客户端的样本
                end_idx = start_idx + n_samples
                user_samples = shuffled_items[start_idx:end_idx]
                
                # 添加到客户端数据中
                federated_test_x[user_id].extend(user_samples)
                testdata_cls_counts[user_id][label] += len(user_samples)
                
                start_idx = end_idx
        
        # 打乱每个客户端的数据顺序
        for user_id in range(num_users):
            np.random.shuffle(federated_test_x[user_id])
            print(f"User {user_id}: {len(federated_test_x[user_id])} test samples")
        
        # 验证分配结果
        total_assigned = sum(len(federated_test_x[i]) for i in range(num_users))
        print(f"Total test samples: {len(test_data)}")
        print(f"Total assigned samples: {total_assigned}")
        print(f"Each user has approximately {total_assigned // num_users} samples")
        
        # 显示前3个客户端的类别分布
        for user_id in range(min(3, num_users)):
            class_count = len(testdata_cls_counts[user_id])
            sample_count = sum(testdata_cls_counts[user_id].values())
            print(f"User {user_id}: {class_count} classes, {sample_count} samples")
        
        return federated_test_x, testdata_cls_counts




