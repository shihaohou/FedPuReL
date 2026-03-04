import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
from Dassl.dassl.utils import set_random_seed, setup_logger
from utils.fed_utils import average_lora_weights, average_weights, evaluate_clients_local



def _tensor_to_cpu(x):
    """Safely detach and move a tensor-like object to CPU."""
    try:
        return x.detach().cpu() if torch.is_tensor(x) else x
    except Exception:
        return x


def state_dict_to_cpu(state_dict, key_filter=None):
    """Move a state_dict to CPU. Optionally keep only keys passing key_filter(key)->bool."""
    if key_filter is None:
        return {k: _tensor_to_cpu(v) for k, v in state_dict.items()}
    return {k: _tensor_to_cpu(v) for k, v in state_dict.items() if key_filter(k)}


def tensor_to_cpu(t):
    """Detach and move a single tensor to CPU."""
    return _tensor_to_cpu(t)

def calculate_accuracy_20(class_accuracy, local_trainer):

    existing_classes = list(class_accuracy.keys())

    head_acc = []
    medium_acc = []
    tail_acc = []
    for cls in existing_classes:
        if cls < len(local_trainer.cls_num_list):
            cls_count = local_trainer.cls_num_list[cls]
            if cls_count > 100:
                head_acc.append(class_accuracy[cls])
            elif 20 < cls_count <= 100:
                medium_acc.append(class_accuracy[cls])
            else:
                tail_acc.append(class_accuracy[cls])

    head_acc_mean = np.mean(head_acc) if head_acc else 0
    medium_acc_mean = np.mean(medium_acc) if medium_acc else 0
    tail_acc_mean = np.mean(tail_acc) if tail_acc else 0
    overall_acc = np.mean(list(class_accuracy.values()))
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Head accuracy (>100 samples): {head_acc_mean:.2f}%")
    print(f"Medium accuracy (20-100 samples): {medium_acc_mean:.2f}%")
    print(f"Tail accuracy (<20 samples): {tail_acc_mean:.2f}%")
    return head_acc_mean, medium_acc_mean, tail_acc_mean, overall_acc




def build_hmt_class_sets(cls_num_list, head_threshold=100, tail_threshold=20):
    """根据每个类别的样本数量构建头/中/尾三段类集合。

    - Head classes: 样本数 > head_threshold (默认 >100)
    - Median classes: tail_threshold < 样本数 <= head_threshold (默认 20~100)
    - Tail classes: 样本数 <= tail_threshold (默认 <=20)
    """
    head_set, mid_set, tail_set = set(), set(), set()

    for cls, count in enumerate(cls_num_list):
        if count > head_threshold:
            head_set.add(cls)
        elif count > tail_threshold:
            mid_set.add(cls)
        else:
            tail_set.add(cls)

    return head_set, mid_set, tail_set


def compute_segment_micro_acc(class_correct, class_total, class_set):
    """基于每类的正确/总数，计算指定类集合的微平均准确率（百分比）。"""
    correct_sum = 0
    total_sum = 0
    for cls in class_set:
        if cls in class_total and class_total[cls] > 0:
            total_sum += class_total[cls]
            correct_sum += class_correct.get(cls, 0)
    if total_sum == 0:
        return 0.0
    return (correct_sum / total_sum) * 100.0


class PeftConfig:
    """统一不同PEFT类型的配置和操作接口"""
    def __init__(self, peft_type, global_trainer, local_trainer, base_global_sd, base_local_sd, datanumber_client, cfg):
        self.peft_type = peft_type
        self.global_trainer = global_trainer
        self.local_trainer = local_trainer
        self.base_global_sd = base_global_sd
        self.base_local_sd = base_local_sd
        self.datanumber_client = datanumber_client
        self.cfg = cfg

        # 根据peft_type初始化特定配置
        if peft_type == "prompt":
            self.param_key = 'prompt_learner.ctx'
            self.param_filter = lambda k: k == 'prompt_learner.ctx'
            self.avg_global_params = copy.deepcopy(base_global_sd[self.param_key])
            self.local_params_mem = {}
        elif peft_type == "lora":
            self.param_key = 'lora_'
            self.param_filter = lambda k: 'lora_' in k
            self.avg_global_params = {k: copy.deepcopy(v) for k, v in base_global_sd.items() if 'lora_' in k}
            self.local_params_mem = {}
        else:  # adapter
            self.param_key = 'img_adap.'
            self.param_filter = lambda k: k.startswith('img_adap.')
            self.avg_global_params = {k: copy.deepcopy(v) for k, v in base_global_sd.items() if k.startswith('img_adap.')}
            self.local_params_mem = {}

    def extract_params(self, state_dict):
        """从state_dict中提取需要的参数"""
        if self.peft_type == "prompt":
            return copy.deepcopy(state_dict[self.param_key])
        else:
            return {k: copy.deepcopy(v) for k, v in state_dict.items() if self.param_filter(k)}

    def load_global_params(self, model):
        """加载全局参数到模型"""
        if self.peft_type == "prompt":
            model.load_state_dict({self.param_key: self.avg_global_params}, strict=False)
        else:
            if len(self.avg_global_params) > 0:
                model.load_state_dict(self.avg_global_params, strict=False)

    def load_local_params(self, model, client_idx):
        """加载本地参数到模型"""
        if client_idx not in self.local_params_mem:
            return False

        if self.peft_type == "prompt":
            model.load_state_dict({self.param_key: self.local_params_mem[client_idx]}, strict=False)
        else:
            if len(self.local_params_mem[client_idx]) > 0:
                model.load_state_dict(self.local_params_mem[client_idx], strict=False)
        return True

    def save_local_params(self, state_dict, client_idx):
        """保存本地参数"""
        self.local_params_mem[client_idx] = self.extract_params(state_dict)

    def aggregate_global_params(self, client_params_dict, idxs_users):
        """聚合全局参数"""
        if self.peft_type == "prompt":
            self.avg_global_params = average_weights(client_params_dict, idxs_users, self.datanumber_client, islist=True)
            self.base_global_sd[self.param_key] = copy.deepcopy(self.avg_global_params)
        elif self.peft_type == "lora":
            self.avg_global_params = average_lora_weights(client_params_dict, idxs_users, self.datanumber_client)
            for k, v in self.avg_global_params.items():
                self.base_global_sd[k] = copy.deepcopy(v)
        else:  # adapter
            total_points = sum([self.datanumber_client[r] for r in idxs_users])
            avg_global_params_new = {}
            for idx in idxs_users:
                fed_w = self.datanumber_client[idx] / total_points if total_points > 0 else 1.0 / max(len(idxs_users), 1)
                for k, v in client_params_dict[idx].items():
                    if k not in avg_global_params_new:
                        avg_global_params_new[k] = v * fed_w
                    else:
                        avg_global_params_new[k] += v * fed_w
            self.avg_global_params = avg_global_params_new
            for k, v in self.avg_global_params.items():
                self.base_global_sd[k] = copy.deepcopy(v)

    def get_trainable_params(self, model):
        """获取可训练参数"""
        if self.peft_type == "prompt":
            return [p for p in model.parameters() if p.requires_grad]
        elif self.peft_type == "lora":
            from loralib.utils import get_lora_parameters
            return list(get_lora_parameters(model))
        else:  # adapter
            return [p for n, p in model.named_parameters() if 'img_adap' in n and p.requires_grad]

    def compute_stage2_loss(self, logits_G, logits_L, labels):
        """计算Stage 2的损失"""
        loss_fn = nn.CrossEntropyLoss()
        logits_sum = logits_G + logits_L

        if self.peft_type in ["prompt", "lora"]:
            # 双损失融合
            loss1 = loss_fn(logits_sum, labels)  # CE(logits_G + logits_L, labels)
            loss2 = loss_fn(logits_L, labels)     # CE(logits_L, labels)
            alpha = self.cfg.OPTIM.FUSION_LOSS_ALPHA
            loss = alpha * loss2 + (1 - alpha) * loss1
        else:  # adapter
            # 仅使用融合logits
            loss = loss_fn(logits_sum, labels)

        return loss


def run_fedpurel_epoch(peft_config, epoch, args, cfg):
    """运行单个FedPuReL训练轮次 (Stage 1 + Stage 2)"""

    # 选择客户端
    if epoch == 0:
        idxs_users = list(range(0, cfg.DATASET.USERS))
    else:
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    print("idxs_users", idxs_users)

    # ============ Stage 1: 全局训练与聚合 ============
    peft_name = {"prompt": "GradPur", "lora": "LoRAGradPur", "adapter": "AdapterGradPur"}[peft_config.peft_type]
    print(f"------------Stage 1: {peft_name} training start epoch:{epoch}-------------")

    global_params_clients = {}
    for idx in idxs_users:
        if peft_config.peft_type == "lora":
            peft_config.global_trainer.set_model_mode("train")

        peft_config.global_trainer.model.load_state_dict(peft_config.base_global_sd, strict=False)
        peft_config.load_global_params(peft_config.global_trainer.model)
        peft_config.global_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)

        gw = peft_config.global_trainer.model.state_dict()
        global_params_clients[idx] = peft_config.extract_params(gw)

    peft_config.aggregate_global_params(global_params_clients, idxs_users)
    print(f"------------Stage 1: {peft_name} aggregation finish epoch:{epoch}-------------")

    # ============ Stage 2: 双logits融合训练 ============
    print(f"------------Stage 2: Dual-logits fusion training start epoch:{epoch}-------------")

    # 准备全局模型
    model_global = peft_config.global_trainer.model
    if peft_config.peft_type == "lora":
        model_global.train()
    model_global.load_state_dict(peft_config.base_global_sd, strict=False)
    peft_config.load_global_params(model_global)
    model_global.eval()
    print(f"Global model prepared with aggregated params")

    # 训练每个客户端的本地模型
    import torch.optim as optim
    for client_idx_pos, idx in enumerate(idxs_users):
        print(f"[Stage 2] Training client {idx} ({client_idx_pos+1}/{len(idxs_users)})...")

        # 加载基础和本地参数
        if peft_config.peft_type == "lora":
            peft_config.local_trainer.set_model_mode("train")

        peft_config.local_trainer.model.load_state_dict(peft_config.base_local_sd, strict=False)
        loaded = peft_config.load_local_params(peft_config.local_trainer.model, idx)
        if loaded:
            print(f"  Loaded previous local params for client {idx}")
        else:
            print(f"  Using base local params for client {idx} (first round)")

        train_loader = peft_config.local_trainer.fed_train_loader_x_dict[idx]
        total_batches = len(train_loader)
        print(f"  Total batches: {total_batches}")

        peft_config.local_trainer.set_model_mode("train")
        model_local = peft_config.local_trainer.model
        model_local.train()

        # 准备优化器
        trainable_params = peft_config.get_trainable_params(model_local)
        optimizer = optim.SGD(trainable_params, lr=cfg.OPTIM.LR)
        param_count = sum(p.numel() for p in trainable_params)
        print(f"  Optimizer initialized with lr={cfg.OPTIM.LR}, # params: {param_count}")

        running_loss = 0.0
        print_freq = max(1, total_batches // 5)

        for batch_idx, batch in enumerate(train_loader):
            images, labels = peft_config.local_trainer.parse_batch_train(batch)

            with torch.no_grad():
                logits_G = model_global(images)

            logits_L = model_local(images)

            loss = peft_config.compute_stage2_loss(logits_G, logits_L, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == total_batches:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{total_batches}], Avg Loss: {avg_loss:.4f}")

        lw = model_local.state_dict()
        peft_config.save_local_params(lw, idx)
        print(f"[Stage 2] Client {idx} training completed. Final avg loss: {running_loss/total_batches:.4f}")

    print(f"------------Stage 2: Dual-logits fusion training finish epoch:{epoch}-------------")

    return idxs_users


def run_personal_test(peft_config, cfg):
    """运行个性化测试 (local + global fusion)"""
    print("------------Personal test (local + global fusion) start-------------")
    all_users = list(range(0, cfg.DATASET.USERS))

    def _eval_personal_fusion(i):
        # 准备全局模型
        model_global = peft_config.global_trainer.model
        if peft_config.peft_type == "lora":
            model_global.train()
        model_global.load_state_dict(peft_config.base_global_sd, strict=False)
        peft_config.load_global_params(model_global)
        model_global.eval()

        # 准备本地模型
        if peft_config.peft_type == "lora":
            peft_config.local_trainer.set_model_mode("train")
        peft_config.local_trainer.model.load_state_dict(peft_config.base_local_sd, strict=False)
        peft_config.load_local_params(peft_config.local_trainer.model, i)
        model_local = peft_config.local_trainer.model
        model_local.eval()

        peft_config.local_trainer.set_model_mode("eval")
        peft_config.local_trainer.evaluator.reset()

        with torch.no_grad():
            data_loader = peft_config.local_trainer.fed_test_loader_x_dict[i]
            for batch in data_loader:
                images, labels = peft_config.local_trainer.parse_batch_test(batch)
                logits = model_global(images) + model_local(images)
                peft_config.local_trainer.evaluator.process(logits, labels)

        results = peft_config.local_trainer.evaluator.evaluate()
        return list(results.values())

    _, _, local_test_acc, local_test_error, local_test_f1 = evaluate_clients_local(
        all_users,
        _eval_personal_fusion,
        None,
        None,
        None
    )

    mean_acc = float(np.mean(local_test_acc)) if local_test_acc else 0.0
    mean_err = float(np.mean(local_test_error)) if local_test_error else 0.0
    mean_f1 = float(np.mean(local_test_f1)) if local_test_f1 else 0.0

    print("Personal test acc:", mean_acc)
    print("Personal test error:", mean_err)
    print("Personal test macro_f1:", mean_f1)
    print("------------Personal test finish-------------")

    return mean_acc, mean_err, mean_f1


def run_global_test(peft_config, epoch):
    """运行全局测试"""
    print("------------Global test (Aggregated global params) start-------------")

    if peft_config.peft_type == "lora":
        peft_config.global_trainer.set_model_mode("train")

    peft_config.global_trainer.model.load_state_dict(peft_config.base_global_sd, strict=False)
    peft_config.load_global_params(peft_config.global_trainer.model)

    result = peft_config.global_trainer.global_test(is_global=True, current_epoch=epoch)

    print("------------Global test finish-------------")
    return result


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
        cfg.DATASET.imagenetROOT = args.imagenetroot
        cfg.DATASET.placesROOT = args.placesroot if hasattr(args, 'placesroot') else args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    
    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.PROMPTFL.CSC = args.csc  # class-specific context
    cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init # initialize words
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROMPTFL.n_general = args.n_general

    # Lora
    cfg.TRAINER.CLIPLORA = CN()
    cfg.TRAINER.CLIPLORA.ENABLED = True
    cfg.TRAINER.CLIPLORA.ENCODER_TYPE = 'vision'  # ['text', 'vision', 'both']
    cfg.TRAINER.CLIPLORA.POSITION = "all"  # ['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3']
    cfg.TRAINER.CLIPLORA.LORA_PARAMS = ['q', 'k', 'v']  # which matrices to apply LoRA  ['q', 'k', 'v']
    cfg.TRAINER.CLIPLORA.RANK = args.lora_rank  # LoRA rank
    cfg.TRAINER.CLIPLORA.ALPHA = cfg.TRAINER.CLIPLORA.RANK*2  # LoRA alpha scaling factor
    cfg.TRAINER.CLIPLORA.DROPOUT = 0.25  # dropout rate
    cfg.TRAINER.CLIPLORA.PREC = "fp16"  # precision ["fp16", "fp32", "amp"]

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.ALPHA = 1.0
    # Align GradPur and PromptFL context length so dual-logits fusion works correctly
    cfg.TRAINER.COOP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # GPLoss
    cfg.LOSS = CN()
    cfg.LOSS.GM = False
    cfg.LOSS.NAME = ""
    cfg.LOSS.ALPHA = 0.
    cfg.LOSS.T = args.T
    cfg.LOSS.LAMBDA = args.lambda_proj


    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client
    cfg.DATASET.IMB_FACTOR = args.imb_factor
    cfg.DATASET.IMB_TYPE = args.imb_type
    cfg.DATASET.USE_MEMORY_CACHE = getattr(args, 'use_memory_cache', False)  # 内存缓存选项
    cfg.DATASET.IID=False

    cfg.OPTIM.ROUND = args.round  # global round
    cfg.OPTIM.MAX_EPOCH = 1  # local epoch
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step
    cfg.OPTIM.LR = args.lr  # learning rate
    cfg.OPTIM.FUSION_LOSS_ALPHA = args.fusion_loss_alpha  # alpha for dual-loss blend in Stage 2

    cfg.MODEL.BACKBONE.PRETRAINED = True

    cfg.DATASET.NUM_CLASSES = args.num_classes



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)
    # print_args(args, cfg)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.trainer != "FedPuReL":
        raise ValueError("Only the FedPuReL trainer is supported after the cleanup. Got: {}".format(args.trainer))

    if args.peft_type not in {"prompt", "lora", "adapter"}:
        raise ValueError("Unsupported peft_type: {}".format(args.peft_type))

    cfg_meta = copy.deepcopy(cfg)
    cfg_meta.defrost()
    if args.peft_type == "prompt":
        cfg_meta.TRAINER.NAME = "PromptFL"
    elif args.peft_type == "lora":
        cfg_meta.TRAINER.NAME = "ClipLora"
    else:
        cfg_meta.TRAINER.NAME = "FedClip"
    cfg_meta.freeze()

    local_trainer = build_trainer(cfg_meta)
    local_trainer.fed_before_train()

    datanumber_client = [len(local_trainer.fed_train_loader_x_dict[i].dataset) for i in range(cfg.DATASET.USERS)]

    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND

    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    local_test_acc_list = []
    local_test_error_list = []
    local_test_f1_list = []
    local_epoch_list = []
    local_time_list = []
    head_acc_list = []
    mid_acc_list = []
    tail_acc_list = []
    start = time.time()
    best_acc = 0
    best_epoch = 0
    last_class_accuracy = []
    best_class_accuracy = []

    if hasattr(local_trainer, 'cls_num_list'):
        _cls_num_list = local_trainer.cls_num_list
    else:
        _cls_num_list = None
        if hasattr(local_trainer, 'dm') and hasattr(local_trainer.dm, 'dataset') and hasattr(local_trainer.dm.dataset, 'y_train'):
            _cls_num_list = [0] * cfg.DATASET.NUM_CLASSES
            for _y in local_trainer.dm.dataset.y_train:
                _cls_num_list[_y] += 1
    head_set, mid_set, tail_set = build_hmt_class_sets(_cls_num_list)
    print(f"H/M/T class sizes (global): {len(head_set)}/{len(mid_set)}/{len(tail_set)}")

    # 根据peft_type构建全局和本地trainer
    trainer_name_map = {
        "prompt": ("GradPur", "PromptFL"),
        "lora": ("LoRAGradPur", "ClipLora"),
        "adapter": ("AdapterGradPur", "FedClip")
    }
    global_name, local_name = trainer_name_map[args.peft_type]

    cfg_global = copy.deepcopy(cfg)
    cfg_local = copy.deepcopy(cfg)
    cfg_global.defrost(); cfg_global.TRAINER.NAME = global_name; cfg_global.freeze()
    cfg_local.defrost(); cfg_local.TRAINER.NAME = local_name; cfg_local.freeze()

    global_trainer = build_trainer(cfg_global)
    local_trainer_unified = build_trainer(cfg_local)
    global_trainer.fed_before_train(is_global=True)
    local_trainer_unified.fed_before_train()

    base_global_sd = copy.deepcopy(global_trainer.model.state_dict())
    base_local_sd = copy.deepcopy(local_trainer_unified.model.state_dict())

    # 创建统一的PeftConfig对象
    peft_config = PeftConfig(
        peft_type=args.peft_type,
        global_trainer=global_trainer,
        local_trainer=local_trainer_unified,
        base_global_sd=base_global_sd,
        base_local_sd=base_local_sd,
        datanumber_client=datanumber_client,
        cfg=cfg
    )

    # 统一的训练循环
    peft_name_display = {"prompt": "Prompt", "lora": "LoRA", "adapter": "Adapter"}[args.peft_type]
    idxs_users = []

    for epoch in range(start_epoch, max_epoch):
        print(f"use FedPuReL-{peft_name_display}: Stage1=Global aggregation, Stage2=Dual-logits fusion training")

        # 运行一个训练轮次 (Stage 1 + Stage 2)
        idxs_users = run_fedpurel_epoch(peft_config, epoch, args, cfg)

        # 个性化测试
        mean_acc, mean_err, mean_f1 = run_personal_test(peft_config, cfg)
        local_time_list.append(time.time() - start)
        local_test_acc_list.append(mean_acc)
        local_test_error_list.append(mean_err)
        local_test_f1_list.append(mean_f1)
        local_epoch_list.append(epoch)

        # 全局测试
        result = run_global_test(peft_config, epoch)
        global_test_acc_list.append(result[0])
        global_test_error_list.append(result[1])
        global_test_f1_list.append(result[2])
        global_epoch_list.append(epoch)
        global_time_list.append(time.time() - start)

        last_class_accuracy = result[3]
        if result[0] > best_acc:
            best_acc = result[0]
            best_class_accuracy = last_class_accuracy

        print("global_test_acc_list:", global_test_acc_list)
        print("maximum test acc:", max(global_test_acc_list))
        print("mean of acc:", np.mean(global_test_acc_list[-5:]))
        print("std of acc:", np.std(global_test_acc_list[-5:]))

        head_acc, medium_acc, tail_acc, _ = calculate_accuracy_20(last_class_accuracy, local_trainer)
        head_acc_list.append(head_acc)
        mid_acc_list.append(medium_acc)
        tail_acc_list.append(tail_acc)

        print("Epoch on server :", epoch)

    # 训练循环结束后的统计和清理部分
    if best_class_accuracy:
        calculate_accuracy_20(best_class_accuracy, local_trainer)

    last_10_epochs = min(10, len(global_test_acc_list))
    if last_10_epochs > 0:
        print("\n----- last {} epoch mean of acc -----".format(last_10_epochs))
        print("Overall accuracy: {:.2f}%".format(np.mean(global_test_acc_list[-last_10_epochs:])))
        print("Head accuracy: {:.2f}%".format(np.mean(head_acc_list[-last_10_epochs:])))
        print("Mid accuracy: {:.2f}%".format(np.mean(mid_acc_list[-last_10_epochs:])))
        print("Tail accuracy: {:.2f}%".format(np.mean(tail_acc_list[-last_10_epochs:])))

    print("\nFinish!")

    # 清理资源
    active_users = idxs_users if idxs_users else range(cfg.DATASET.USERS)

    try:
        peft_config.global_trainer.fed_after_train()
    except Exception:
        pass

    for _ in active_users:
        try:
            peft_config.local_trainer.fed_after_train()
        except Exception:
            pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fedavg", choices=["fedavg"], help="aggregation model (FedPuReL only supports fedavg)")
    parser.add_argument("--trainer", type=str, default="FedPuReL", choices=["FedPuReL"], help="only the unified FedPuReL trainer is available")
    parser.add_argument("--peft_type", type=str, default="prompt", choices=["prompt", "lora", "adapter"], help="PEFT type for FedPuReL")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.4, help='the fraction of clients: C')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test_acpfl batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # parameters of datasets
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir',help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.05,help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor，IF = 100, 50 and 10')


    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=2, help="number of prompts")
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to aggregate")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")
    parser.add_argument('--csc', default="True", help="is using the class-specific context")
    


    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--imagenetroot", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--placesroot", type=str, default="./DATA/", help="path to places dataset")
    parser.add_argument("--output-dir", type=str, default="output/test/", help="output directory")
    parser.add_argument("--config-file", type=str, default="configs/trainers/PromptFL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar10_LT.yaml",help="path to config file for dataset setup")  #############
    parser.add_argument("--resume", type=str, default=None,help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,help="modify config options using the command-line")

    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float, metavar='LR', help='initial learning rate',dest='lr')
    parser.add_argument('--schedule', default=[6, 10], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--dataset', default="imagenet_LT")
    parser.add_argument('--visualize_interval', type=int, default=10, help="Interval for generating visualizations")
    parser.add_argument('--n_general', type=int, default=1, help="number of text encoder of text prompts")
    parser.add_argument('--n_disclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--n_simclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--prompt_depth', type=int, default=9)
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=8, help='Low-rank dimension for LoRA')

    # GradPur arguments
    parser.add_argument('--lambda_proj', type=float, default=1, help='Projection loss weight for GradPur')
    parser.add_argument('--T', type=float, default=1.0, help='Temperature parameter for GradPur loss')


    # ResFed Dual-loss fusion alpha for Stage 2 (loss = alpha*loss2 + (1-alpha)*loss1)
    parser.add_argument('--fusion_loss_alpha', type=float, default=0.99, help='alpha for dual-loss blending in Stage 2 training')

    args = parser.parse_args()
    main(args)
