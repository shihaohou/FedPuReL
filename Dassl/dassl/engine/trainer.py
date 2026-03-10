import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from Dassl.dassl.modeling import build_head, build_backbone
from Dassl.dassl.evaluation import build_evaluator
import os
import copy

from collections import defaultdict

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features
        # print("self.backbone",self.backbone)

        self.head = None
        # print("model_cfg.HEAD.NAME",model_cfg.HEAD.NAME)
        # print("model_cfg.HEAD.HIDDEN_LAYERS",model_cfg.HEAD.HIDDEN_LAYERS)
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:

            print("num_classes",num_classes)
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            print("save model name",name)
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print(f"Load {model_path} to {name} (epoch={epoch})")
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test_acpfl", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, idx=-1,global_epoch=-1,is_fed=False,global_weight=None, fedprox=False, mu=0.5):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train(is_fed)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.sched.step()
            self.before_epoch()
            self.run_epoch(idx, global_epoch, global_weight=global_weight,fedprox=fedprox,mu=mu)
            self.after_epoch()
        self.after_train(idx,global_epoch,is_fed)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def GradPur_backward_and_update(
        self, loss_a, loss_b, lambda_=1, names=None
    ):
        # loss_b not increase is okay
        # loss_a has to decline
        self.model_zero_grad(names)
        # get name of the model parameters
        names = self.get_model_names(names)
        # backward loss_a
        self.detect_anomaly(loss_b)
        loss_b.backward(retain_graph=True)
        # normalize gradient
        b_grads = []
        for name in names:
            for p in self._models[name].parameters():
                b_grads.append(p.grad.clone())

        # optimizer don't step
        for name in names:
            self._optims[name].zero_grad()

        # backward loss_a
        self.detect_anomaly(loss_a)
        loss_a.backward()
        for name in names:
            for p, b_grad in zip(self._models[name].parameters(), b_grads):
                # calculate cosine distance
                b_grad_norm = b_grad / torch.linalg.norm(b_grad)
                a_grad = p.grad.clone()
                a_grad_norm = a_grad / torch.linalg.norm(a_grad)

                if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
                    p.grad = a_grad - lambda_ * torch.dot(
                        a_grad.flatten(), b_grad_norm.flatten()
                    ) * b_grad_norm

        # optimizer
        for name in names:
            self._optims[name].step()

class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        dm = DataManager(self.cfg)

        # self.train_loader_x = dm.train_loader_x
        # self.train_loader_u = dm.train_loader_u  # optional, can be None
        # self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.fed_train_loader_x_dict = dm.fed_train_loader_x_dict
        self.fed_test_loader_x_dict = dm.fed_test_loader_x_dict

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.classnames = dm.classnames
        self.client_proportion = dm.client_proportion

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        print("self.num_classes", self.num_classes)
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            # self.model = nn.DataParallel(self.model)

    def train(self,idx=-1,global_epoch=0,is_fed=False,global_weight=None, fedprox=False, mu=0.5):
        super().train(self.start_epoch, self.max_epoch,idx,global_epoch,is_fed,global_weight,fedprox,mu)

    def fed_before_train(self,is_global = False):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        # self.start_epoch = self.resume_model_if_exist(directory)
        self.start_epoch = 0

        # Initialize summary writer
        if is_global:

            writer_dir = osp.join(self.output_dir, "global/tensorboard")
        else:
            writer_dir = osp.join(self.output_dir, "local/tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        self.total_time_start = time.time()

    def before_train(self, is_fed=False):
        if not is_fed:
            directory = self.cfg.OUTPUT_DIR
            if self.cfg.RESUME:
                directory = self.cfg.RESUME
            # self.start_epoch = self.resume_model_if_exist(directory)
        self.start_epoch = 0

        # Initialize summary writer
        if not is_fed:
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self,idx=-1,epoch=0,is_fed=False):
        print("Finish training:",idx,"user")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")

                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")

            self.test(idx=idx,current_epoch=epoch)


        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        if not is_fed:
            print(f"Total time Elapsed: {elapsed}")
        else:
            print(f"{idx} User, Elapsed: {elapsed}")

        # Close writer
        if not is_fed:
            self.close_writer()

    def fed_after_train(self):
        elapsed = round(time.time() - self.total_time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Total time Elapsed: {elapsed}")
        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                # self.save_model(
                #     self.epoch,
                #     self.output_dir,
                #     model_name="model-best.pth.tar"
                # )

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"  # in case val_loader is None
            # data_loader = self.test_loader
            data_loader = self.fed_test_loader_x_dict[idx]

        print(f"Evaluate on the client{idx}_{split} set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        if not is_global and idx < 0:
            current_epoch = self.epoch
            # print("current epoch", current_epoch)
        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)
            # print("tag",tag,"value:",v, ",current_epoch:",current_epoch)

        return list(results.values())

    # def global_test(self, split=None, is_global=False, current_epoch=0, idx=-1):  # 原始
    #     """A generic testing pipeline."""
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()
    #
    #     if split is None:
    #         split = self.cfg.TEST.SPLIT
    #
    #     if split == "val" and self.val_loader is not None:
    #         data_loader = self.val_loader
    #     else:
    #         split = "test_acpfl"  # in case val_loader is None
    #         data_loader = self.test_loader
    #
    #     print(f"Evaluate on the *{split}* set")
    #
    #     for batch_idx, batch in enumerate(tqdm(data_loader)):
    #         input, label = self.parse_batch_test(batch)
    #         output = self.model_inference(input)
    #         self.evaluator.process(output, label)
    #
    #     results = self.evaluator.evaluate()
    #     if not is_global and idx < 0:
    #         current_epoch = self.epoch
    #         # print("current epoch", current_epoch)
    #     for k, v in results.items():
    #         tag = f"{split}/{k}"
    #         if not is_global:
    #             tag = f"{tag}/{str(idx)}"
    #         self.write_scalar(tag, v, current_epoch)
    #         # print("tag",tag,"value:",v, ",current_epoch:",current_epoch)
    #
    #     return list(results.values())




    def test_acclist(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"  # in case val_loader is None
            data_loader = self.fed_test_loader_x_dict[idx] if idx >= 0 else self.test_loader

        print(f"Evaluate on the {'global' if global_test else f'client{idx}'} {split} set")

        # 创建字典来存储每个类别的预测结果
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, label)

                # 计算每个类别的正确预测和总数
                pred = output.argmax(dim=1)
                for l, p in zip(label, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        # 计算每个类别的准确度
        class_accuracy = {cls: (correct / max(total, 1)) * 100
                          for cls, correct, total in zip(class_correct.keys(),
                                                         class_correct.values(),
                                                         class_total.values())}

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        # 将整体结果和类别准确度一起返回
        return list(results.values()), class_accuracy


    def local_test(self, split=None, is_global=False, current_epoch=0, idx=-1):
        self.set_model_mode("eval")
        self.evaluator.reset()

        split = "test_acpfl"  # in case val_loader is None
        data_loader = self.fed_test_loader_x_dict[idx]

        print(f"Evaluate on the client{idx} test set")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, labels = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, labels)
                pred = output.argmax(dim=1)
                for l, p in zip(labels, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100
                          for cls in class_correct.keys()}

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        # 返回额外的统计信息用于准确计算
        return list(results.values()) + [class_accuracy, class_correct, class_total]


    def global_test(self, split=None, is_global=False, current_epoch=0, idx=-1):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, labels = self.parse_batch_test(batch)
                output = self.model_inference(input)
                self.evaluator.process(output, labels)
                pred = output.argmax(dim=1)
                for l, p in zip(labels, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100
                          for cls in class_correct.keys()}

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        # 返回额外的统计信息用于准确计算
        return list(results.values()) + [class_accuracy, class_correct, class_total]



    def test_fedpgp(self, split=None, is_global=False, current_epoch=0, idx=-1):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "local":
            data_loader = self.fed_test_loader_x_dict[idx]
        else:
            split = "test_acpfl"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)


        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, labels = self.parse_batch_test(batch)
                # output = self.model_inference(input)
                output =self.model(input, training=False)
                # print("Output type:", type(output))
                # print("Output:", output)
                self.evaluator.process(output, labels)
                pred = output.argmax(dim=1)
                for l, p in zip(labels, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100
                          for cls in class_correct.keys()}

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        return list(results.values()) + [class_accuracy, class_correct, class_total]


    def global_test_fedpgp(self, split=None, is_global=False, current_epoch=0, idx=-1):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)


        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, labels = self.parse_batch_test(batch)
                # output = self.model_inference(input)
                output =self.model(input, training=False)
                # print("Output type:", type(output))
                # print("Output:", output)
                self.evaluator.process(output, labels)
                pred = output.argmax(dim=1)
                for l, p in zip(labels, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100
                          for cls in class_correct.keys()}

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        return list(results.values()) + [class_accuracy, class_correct, class_total]

    def test_afpcl(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        """A generic testing pipeline for AFPCL."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"
            data_loader = self.fed_test_loader_x_dict[idx]

        print(f"Evaluate on the client{idx}_{split} set")

        # 确保使用正确的 AdaptiveFederatedPromptLoss
        if is_global:
            self.model.adaptive_loss = copy.deepcopy(self.adaptive_loss)
        else:
            # 使用客户端特定的 AdaptiveFederatedPromptLoss，如果有的话
            pass  # 可能需要添加逻辑来加载或使用客户端特定的 AdaptiveFederatedPromptLoss

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                ce_logits, adaptive_logits = self.model(input, label)

                # 组合 CE 和 AFPCL logits
                combined_logits = ce_logits + self.cfg.TRAINER.PROMPTFL.ADAPTIVE_WEIGHT * adaptive_logits

                self.evaluator.process(combined_logits, label)

        results = self.evaluator.evaluate()

        if not is_global and idx < 0:
            current_epoch = self.epoch

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        return list(results.values())

    def global_test_afpcl(self, split=None, is_global=True, current_epoch=0, idx=-1):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test_acpfl"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                ce_logits, adaptive_logits = self.model(input, label)
                combined_logits = ce_logits + self.cfg.TRAINER.PROMPTFL.ADAPTIVE_WEIGHT * adaptive_logits
                output = combined_logits
                self.evaluator.process(output, label)

                pred = output.argmax(dim=1)
                for l, p in zip(label, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

        results = self.evaluator.evaluate()

        class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100
                          for cls in class_correct.keys()}

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global:
                tag = f"{tag}/{str(idx)}"
            self.write_scalar(tag, v, current_epoch)

        # Print overall accuracy and per-class accuracy
        # print(f"Overall accuracy: {results['accuracy']:.2f}%")
        # print("Per-class accuracy:")
        # for cls, acc in class_accuracy.items():
            # print(f"Class {cls}: {acc:.2f}%")

        return list(results.values()) + [class_accuracy, class_correct, class_total]

    def model_inference(self, input):
        return self.model(input)
    # def model_inference(self, input):
    #     ce_logits, contrast_logits = self.model(input)
    #     combined_logits = ce_logits + self.cfg.TRAINER.PROMPTFL.PROCO_WEIGHT * contrast_logits
    #     return combined_logits

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self, idx=-1, global_epoch=-1, global_weight=None, fedprox=False, mu=0.5):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if idx>=0:
            loader = self.fed_train_loader_x_dict[idx]
        else:
            loader = self.train_loader_x
        self.num_batches = len(loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(loader):
            data_time.update(time.time() - end)
            if fedprox:
                loss_summary = self.forward_backward(batch, global_weight=global_weight, fedprox=fedprox, mu=mu)
            else:
                loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                info += [f"user {idx}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            if global_epoch >= 0:
                max_per_epoch = self.max_epoch*self.num_batches
                # print("max_per_epoch",max_per_epoch)
                n_iter = global_epoch*max_per_epoch + n_iter
                # print("n_iter",n_iter)
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name + "/" + str(idx), meter.avg, n_iter)
                # print("name:",name,",value:",meter.avg, ",n_iter:",n_iter)
            self.write_scalar("train/lr/" + str(idx), self.get_current_lr(), n_iter)
            # print("name: lr", ",value:", self.get_current_lr(), ",n_iter:", n_iter)

            end = time.time()


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
