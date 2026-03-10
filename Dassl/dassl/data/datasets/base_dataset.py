import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import numpy as np
from collections import defaultdict, Counter

from Dassl.dassl.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, federated_train_x=None, train_u=None, val=None, federated_test_x=None, test=None):
        self._train_x = train_x  # labeled training data
        self._federated_train_x = federated_train_x # federated labeled training data (optional)
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._federated_test_x = federated_test_x  # federated labeled test_acpfl data
        self._test = test # test_acpfl data
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def federated_train_x(self):
        return self._federated_train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def federated_test_x(self):
        return self._federated_test_x

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        assert len(source_domains) > 0, "source_domains (list) is empty"
        assert len(target_domains) > 0, "target_domains (list) is empty"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []
        print("data_sources len",len(data_sources))

        for data_source in data_sources:
            print("data_source len", len(data_source))
            tracker = self.split_dataset_by_label(data_source)
            print("tracker len",len(tracker))
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


    def generate_federated_fewshot_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate = 0.0, repeat=False
    ):
        """Generate a federated few-shot dataset (typically for the federated training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            num_users (int): number of users
            repeat (bool): repeat images if needed (default: False).
        Return:
            Directory[list]:list of data for each user
        """
        print(f"Creating a {num_shots}-shot federated dataset")
        output_dict = defaultdict(list)
        if num_shots < 1:
            for idx in range(num_users):
                if len(data_sources) == 1:
                    output_dict[idx] = data_sources[0]
                output_dict[idx].append(data_sources)

        else:
            user_class_dict = defaultdict(list)
            class_num = self.get_num_classes(data_sources[0])
            print("class_num",class_num)
            class_per_user = int(round(class_num/num_users))
            class_list = list(range(0,class_num))
            random.seed(2023)
            random.shuffle(class_list)
            if repeat_rate > 0:
                repeat_num = int(repeat_rate*class_num)
                class_repeat_list = class_list[0:repeat_num]
                class_norepeat_list = class_list[repeat_num:class_num]
                class_per_user = int(round((class_num-repeat_num)/num_users))
                fold = int(num_users/num_shots)
                print("repeat_num",repeat_num)
                print("class_repeat_list", class_repeat_list)
                print("class_norepeat_list",class_norepeat_list)
                print("fold", fold)
                if fold > 0:
                    client_idx_fold = defaultdict(list)
                    client_per_fold = int(round(num_users/fold))
                    repeat_per_fold = int(round(repeat_num/fold))
                    client_list = list(range(0,num_users))
                    random.shuffle(client_list)
                    for i in range(fold):
                        client_idx_fold[i] = client_list[i*client_per_fold:min((i + 1) * client_per_fold, num_users)]


            for data_source in data_sources:
                tracker = self.split_dataset_by_label(data_source)

                for idx in range(num_users):
                    if is_iid:
                        user_class_dict[idx] = list(range(0, class_num))
                    else:
                        if repeat_rate == 0.0:
                            # user_class_dict[idx] = list(range(idx*class_per_user,min((idx+1)*class_per_user,class_num)))
                            if idx == num_users-1:
                                user_class_dict[idx] = class_list[idx * class_per_user:  class_num]
                            else:
                                user_class_dict[idx] = class_list[idx * class_per_user: (idx + 1) * class_per_user]
                        else:
                            user_class_dict[idx] = []
                            if fold > 0:
                                for k, v in client_idx_fold.items():
                                    if idx in v:
                                        if k == len(client_idx_fold) - 1:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold: repeat_num])
                                        else:
                                            user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold:(k + 1) * repeat_per_fold])
                            else:
                                user_class_dict[idx].extend(class_repeat_list)
                            print("user_class_dict repeat part", user_class_dict[idx])

                            if idx == num_users - 1:
                                user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user:  class_num - repeat_num])
                                print("user_class_dict nonrepeat part", class_norepeat_list[idx * class_per_user:  class_num - repeat_num])
                            else:
                                user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])
                                print("user_class_dict nonrepeat part", class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])

                    print("user class dict total",user_class_dict[idx])

                    dataset = []

                    for label, items in tracker.items():
                        if label in user_class_dict[idx]:
                            if repeat_rate == 0.0:
                                if len(items) >= num_shots:
                                    sampled_items = random.sample(items, num_shots)
                                else:
                                    if repeat:
                                        sampled_items = random.choices(items, k=num_shots)
                                    else:
                                        sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                if label in class_repeat_list:
                                    if int(num_shots/num_users) >0:
                                        tmp_num_shots = int(num_shots/num_users)
                                    else:
                                        tmp_num_shots = 1
                                    sampled_items = random.sample(items, tmp_num_shots)
                                else:
                                    sampled_items = random.sample(items, num_shots)
                                dataset.extend(sampled_items)

                    output_dict[idx] = dataset
                    print("idx:",idx,",","output_dict_len:",len(output_dict[idx]))

        return output_dict


    def generate_federated_dataset(
        self, *data_sources, num_shots=-1, num_users=5, is_iid=False, repeat_rate = 0.0, repeat=False
    ):
        """Generate a federated dataset (typically for the federated baseline training set).
        Every client owns total number of class per client

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            num_users (int): number of users
            repeat (bool): repeat images if needed (default: False).
        Return:
            Directory[list]:list of data for each user
        """
        print(f"Creating a baseline federated dataset")
        output_dict = defaultdict(list)
        user_class_dict = defaultdict(list)
        sample_per_user = defaultdict(int)
        sample_order = defaultdict(list)
        class_num = self.get_num_classes(data_sources[0])
        print("class_num",class_num)
        class_per_user = int(round(class_num/num_users))
        class_list = list(range(0, class_num))
        random.seed(1)
        random.shuffle(class_list)
        if repeat_rate > 0:
            repeat_num = int(repeat_rate * class_num)
            class_repeat_list = class_list[0:repeat_num]
            class_norepeat_list = class_list[repeat_num:class_num]
            class_per_user = int(round((class_num - repeat_num) / num_users))
            if repeat_rate > 0:
                repeat_num = int(repeat_rate*class_num)
                class_repeat_list = class_list[0:repeat_num]
                class_norepeat_list = class_list[repeat_num:class_num]
                class_per_user = int(round((class_num-repeat_num)/num_users))
                fold = int(num_users/num_shots)
                print("repeat_num",repeat_num)
                print("class_repeat_list", class_repeat_list)
                print("class_norepeat_list",class_norepeat_list)
                print("fold", fold)
                if fold > 0:
                    client_idx_fold = defaultdict(list)
                    client_per_fold = int(round(num_users/fold))
                    repeat_per_fold = int(round(repeat_num/fold))
                    client_list = list(range(0,num_users))
                    random.shuffle(client_list)
                    for i in range(fold):
                        client_idx_fold[i] = client_list[i*client_per_fold:min((i + 1) * client_per_fold, num_users)]

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            for label, items in tracker.items():
                sample_order[label] = list(range(0, len(items)))
                sample_per_user[label] = int(round(len(items) / num_users))
                # print("label, sample_per_user",label, sample_per_user[label])
                random.shuffle(sample_order[label])
                if repeat_rate > 0 and fold > 0:
                    sample_per_user[label] = int(round(len(items) /(num_users/fold)))

            for idx in range(num_users):
                if is_iid:
                    user_class_dict[idx] = list(range(0, class_num))
                else:
                    if repeat_rate == 0.0:
                        # user_class_dict[idx] = list(range(idx*class_per_user,min((idx+1)*class_per_user,class_num)))
                        if idx == num_users - 1:
                            user_class_dict[idx] = class_list[idx * class_per_user:  class_num]
                        else:
                            user_class_dict[idx] = class_list[idx * class_per_user: (idx + 1) * class_per_user]
                    else:
                        user_class_dict[idx] = []
                        if fold > 0:
                            for k,v in client_idx_fold.items():
                                if idx in v:
                                    if k == len(client_idx_fold)-1:
                                        user_class_dict[idx].extend(class_repeat_list[k*repeat_per_fold: repeat_num])
                                    else:
                                        user_class_dict[idx].extend(class_repeat_list[k * repeat_per_fold:(k + 1) * repeat_per_fold])
                        else:
                            user_class_dict[idx].extend(class_repeat_list)
                        print("user_class_dict repeat part",user_class_dict[idx])

                        if idx == num_users - 1:
                            user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user:  class_num-repeat_num])
                            print("user_class_dict nonrepeat part", class_norepeat_list[idx * class_per_user:  class_num-repeat_num])
                        else:
                            user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])
                            print("user_class_dict nonrepeat part", class_norepeat_list[idx * class_per_user: (idx + 1) * class_per_user])

                print("user class dict total",user_class_dict[idx])

            # for idx in range(num_users):
            #     if is_iid:
            #         user_class_dict[idx] = list(range(0, class_num))
            #     else:
            #         if repeat_rate == 0:
            #             # user_class_dict[idx] = list(range(idx*class_per_user,min((idx+1)*class_per_user,class_num)))
            #             user_class_dict[idx] = class_list[idx * class_per_user: min((idx + 1) * class_per_user, class_num)]
            #         else:
            #             user_class_dict[idx] = []
            #             user_class_dict[idx].extend(class_repeat_list)
            #             user_class_dict[idx].extend(class_norepeat_list[idx * class_per_user: min((idx + 1) * class_per_user, class_num - repeat_num)])


                dataset = []

                for label, items in tracker.items():
                    if label in user_class_dict[idx]:
                        if is_iid:
                            sampled_items=[]
                            for k,v in enumerate(items):
                                if k in sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))]:
                                    # print(idx,label,sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))])
                                    sampled_items.append(v)
                            dataset.extend(sampled_items)
                        else:
                            if repeat_rate == 0.0:
                                sampled_items = items
                                dataset.extend(sampled_items)
                            else:
                                if label in user_class_dict[idx][0:repeat_num]:
                                    sampled_items = []
                                    for k, v in enumerate(items):
                                        if k in sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label],len(items))]:
                                            # print(idx,label,sample_order[label][idx * sample_per_user[label]: min((idx + 1) * sample_per_user[label], len(items))])
                                            sampled_items.append(v)
                                    dataset.extend(sampled_items)
                                else:
                                    sampled_items = items
                                    dataset.extend(sampled_items)


                output_dict[idx] = dataset
                print("idx:",idx,",","output_dict_len:",len(output_dict[idx]))

        return output_dict

    def generate_federated_dataset_imagenet(self, data_source, num_users=5, is_iid=False, beta=0.5):
        print(f"Creating a federated dataset with {'IID' if is_iid else 'non-IID'} distribution")

        # Split dataset by label
        tracker = self.split_dataset_by_label(data_source)
        class_num = len(tracker)

        output_list = [[] for _ in range(num_users)]
        traindata_cls_counts = {i: defaultdict(int) for i in range(num_users)}

        if is_iid:
            all_indices = list(range(len(data_source)))
            np.random.shuffle(all_indices)
            chunk_size = len(all_indices) // num_users
            for i in range(num_users):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_users - 1 else len(all_indices)
                user_data = [data_source[idx] for idx in all_indices[start_idx:end_idx]]
                output_list[i] = user_data
                traindata_cls_counts[i] = Counter(item.label for item in user_data)
        else:
            for label, items in tracker.items():
                idx_k = list(range(len(items)))
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_users))
                proportions = proportions / proportions.sum()
                splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = np.split(idx_k, splits)

                for j, idx in enumerate(idx_batch):
                    user_data = [items[i] for i in idx]
                    output_list[j].extend(user_data)
                    traindata_cls_counts[j][label] += len(user_data)

        # Shuffle data for each user
        for i in range(num_users):
            np.random.shuffle(output_list[i])
            print(f"User {i}: {len(output_list[i])} samples")

        return output_list, traindata_cls_counts

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output
