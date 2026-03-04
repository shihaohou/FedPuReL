import torch
import copy
from prettytable import PrettyTable
import torch.nn.functional as F
import math
import numpy as np

def average_weights(w,idxs_users,datanumber_client,islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    
    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points
        
        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg

def average_lora_weights(local_sd_list, idxs_users, datanumber_client):
    # local_sd_list[i] 是客户端 i 的 state_dict
    # 只聚合带 "lora_" 的参数
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    w_avg = None
    for idx in idxs_users:
        sd = local_sd_list[idx]
        fed_freq = datanumber_client[idx] / total_data_points
        if w_avg is None:
            # 初始化
            w_avg = {}
            for k, v in sd.items():
                if "lora_" in k:
                    w_avg[k] = v * fed_freq
        else:
            for k, v in sd.items():
                if k in w_avg:
                    w_avg[k] += v * fed_freq
    return w_avg


def is_lora_A_key(key: str) -> bool:
    """判断是否为 LoRA 的 A 矩阵参数键。

    兼容常见命名：
    - peft/LoRA: "lora_A"
    - diffusers 等实现："lora_down"
    """
    if "lora_" not in key:
        return False
    k = key.lower()
    return ("lora_a" in k) or ("lora_down" in k)


def is_lora_B_key(key: str) -> bool:
    """判断是否为 LoRA 的 B 矩阵参数键。

    兼容常见命名：
    - peft/LoRA: "lora_B"
    - diffusers 等实现："lora_up"
    """
    if "lora_" not in key:
        return False
    k = key.lower()
    return ("lora_b" in k) or ("lora_up" in k)


def average_lora_A_only(local_sd_list, idxs_users, datanumber_client):
    """仅对 LoRA 的 A 矩阵进行联邦加权平均聚合。

    Args:
        local_sd_list: List[state_dict]，其中 local_sd_list[i] 是客户端 i 的模型 state_dict
        idxs_users: 本轮参与的客户端索引列表
        datanumber_client: 每个客户端的样本数列表

    Returns:
        一个仅包含 LoRA-A 相关键的聚合字典（可用于 strict=False 方式加载到模型中）
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])
    w_avg = {}
    for idx in idxs_users:
        sd = local_sd_list[idx]
        fed_freq = datanumber_client[idx] / total_data_points if total_data_points > 0 else 1.0 / max(len(idxs_users), 1)
        for k, v in sd.items():
            if is_lora_A_key(k):
                if k not in w_avg:
                    w_avg[k] = v * fed_freq
                else:
                    w_avg[k] += v * fed_freq
    return w_avg

    
def average_weights_F(w, idxs_users, datanumber_client):
    """
    对模型权重或耦合函数参数进行加权平均
    """
    w_avg = copy.deepcopy(w[idxs_users[0]])
    total = sum(datanumber_client[i] for i in idxs_users)

    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * datanumber_client[idxs_users[0]] / total
        for i in range(1, len(idxs_users)):
            w_avg[key] += w[idxs_users[i]][key] * datanumber_client[idxs_users[i]] / total

    return w_avg

def average_weights_afpcl(w, idxs_users, datanumber_client, islist=False):
    """
    Returns the average of the weights.
    """
    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    adaptive_loss_params = {}

    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = [wi * fed_avg_freqs for wi in w_avg]
            else:
                w_avg = [w_avg[i] + wi * fed_avg_freqs for i, wi in enumerate(w[idxs_users[idx]])]
        else:
            if idx == 0:
                for key in w_avg:
                    if key.startswith('adaptive_loss.'):
                        adaptive_loss_params[key] = w_avg[key] * fed_avg_freqs
                    else:
                        w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    if key.startswith('adaptive_loss.'):
                        if key in adaptive_loss_params:
                            adaptive_loss_params[key] += w[idxs_users[idx]][key] * fed_avg_freqs
                        else:
                            adaptive_loss_params[key] = w[idxs_users[idx]][key] * fed_avg_freqs
                    else:
                        w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    # Merge adaptive_loss parameters back into w_avg
    if not islist:
        w_avg.update(adaptive_loss_params)

    return w_avg


def count_parameters(model,model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:
            # if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def compute_softmax_weights(v_list, v_zs, tau=0.1):
    """Compute softmax weights from cosine similarity between client proxies and server proxy.

    Args:
        v_list: List[Tensor] each of shape (d,), client proxies.
        v_zs: Tensor shape (d,), server zero-shot proxy.
        tau: temperature for softmax.
    Returns:
        Tensor shape (len(v_list),) normalized weights.
    """
    device = v_list[0].device if isinstance(v_list[0], torch.Tensor) else torch.device('cpu')
    vz = v_zs.to(device)
    sims = []
    for v in v_list:
        v = v.to(device)
        denom = (v.norm(p=2) * vz.norm(p=2)).clamp_min(1e-12)
        sims.append(torch.dot(v, vz) / denom)
    sims = torch.stack(sims)  # (M,)
    w = F.softmax(sims / max(tau, 1e-6), dim=0)
    return w


def aggregate_prompt_deltas(delta_state_list, weights):
    """Aggregate prompt deltas using provided weights.

    Args:
        delta_state_list: list of dicts mapping param_name -> tensor (e.g., 'prompt_learner.ctx').
        weights: Tensor of shape (M,) or list of floats summing to 1.
    Returns:
        dict mapping param_name -> aggregated tensor.
    """
    if not delta_state_list:
        return {}
    # Ensure weights tensor
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)

    # Initialize aggregated dict with zeros
    agg = {}
    keys = list(delta_state_list[0].keys())
    for k in keys:
        agg[k] = torch.zeros_like(delta_state_list[0][k])

    for w, d in zip(weights, delta_state_list):
        for k in keys:
            agg[k] += w * d[k]
    return agg


def _compute_segment_micro_acc(class_correct, class_total, class_set):
    """基于每类的正确/总数，计算指定类集合的微平均准确率（百分比）。"""
    if class_correct is None or class_total is None or class_set is None:
        return 0.0, False
    correct_sum = 0
    total_sum = 0
    for cls in class_set:
        if cls in class_total and class_total[cls] > 0:
            total_sum += class_total[cls]
            correct_sum += class_correct.get(cls, 0)
    if total_sum == 0:
        return 0.0, False
    return (correct_sum / total_sum) * 100.0, True


def evaluate_clients_local(client_indices, eval_fn, head_set=None, mid_set=None, tail_set=None, print_title=None):
    """
    通用的客户端本地评估封装：
    - 逐客户端调用 eval_fn(idx) 获取结果 res
    - 期望 res 至少包含 [acc, error, f1]，若包含末尾两个字典 [class_correct, class_total]，则计算 H/M/T 微平均准确率

    Args:
        client_indices: 可迭代的客户端索引
        eval_fn: 可调用对象，签名为 eval_fn(idx) -> list/tuple，至少 [acc, error, f1]
        head_set/mid_set/tail_set: 类别集合（可选），用于计算微平均准确率
        print_title: 若提供，将打印每客户端的 H/M/T 统计以及均值

    Returns:
        (local_results, post_agg_hmt, acc_list, err_list, f1_list)
        其中 post_agg_hmt[cid] = (h, m, t, has_h, has_m, has_t)
    """
    local_results = []
    post_agg_hmt = {}
    acc_list, err_list, f1_list = [], [], []

    # 临时存放用于求均值的 H/M/T
    post_h_list, post_m_list, post_t_list = [], [], []

    for cid in client_indices:
        res = eval_fn(cid)
        local_results.append(res)
        if len(res) >= 3:
            acc_list.append(res[0])
            err_list.append(res[1])
            f1_list.append(res[2])

        class_correct = res[-2] if len(res) >= 2 and isinstance(res[-2], dict) else None
        class_total = res[-1] if len(res) >= 1 and isinstance(res[-1], dict) else None

        h_val, has_h = _compute_segment_micro_acc(class_correct, class_total, head_set)
        m_val, has_m = _compute_segment_micro_acc(class_correct, class_total, mid_set)
        t_val, has_t = _compute_segment_micro_acc(class_correct, class_total, tail_set)
        post_agg_hmt[cid] = (h_val, m_val, t_val, has_h, has_m, has_t)

        if has_h:
            post_h_list.append(h_val)
        if has_m:
            post_m_list.append(m_val)
        if has_t:
            post_t_list.append(t_val)

    if print_title is not None:
        print(print_title)
        for cid in client_indices:
            h_a, m_a, t_a, has_h_a, has_m_a, has_t_a = post_agg_hmt[cid]
            h_str = f"{h_a:.2f}" if has_h_a else "None"
            m_str = f"{m_a:.2f}" if has_m_a else "None"
            t_str = f"{t_a:.2f}" if has_t_a else "None"
            print(f"client {cid}:[head:{h_str}, mid:{m_str}, tail:{t_str}]")

        if len(post_h_list) > 0 or len(post_m_list) > 0 or len(post_t_list) > 0:
            avg_h_a = float(np.mean(post_h_list)) if len(post_h_list) > 0 else 0.0
            avg_m_a = float(np.mean(post_m_list)) if len(post_m_list) > 0 else 0.0
            avg_t_a = float(np.mean(post_t_list)) if len(post_t_list) > 0 else 0.0
            print(f"Avg:[head:{avg_h_a:.2f}, mid:{avg_m_a:.2f}, tail:{avg_t_a:.2f}]")

    return local_results, post_agg_hmt, acc_list, err_list, f1_list


def fused_local_test_by_confidence(trainer, idx, robust_ctx, personal_ctx):
    """
    FedGrad: 基于置信度的决策级选择融合测试（复用 PromptFL/GradPur 风格接口）。
    - 模型共享 image features，两路 text/prompt logits 比较置信度，冲突时取更高置信度一方。

    Returns: list(results.values()) + [class_accuracy, class_correct, class_total]
    """
    model = trainer.model
    sd = model.state_dict()
    orig_ctx = sd['prompt_learner.ctx']
    try:
        trainer.set_model_mode("eval")
        trainer.evaluator.reset()
        model.eval()
        with torch.no_grad():
            data_loader = trainer.fed_test_loader_x_dict[idx]
            from collections import defaultdict
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            for batch_idx, batch in enumerate(data_loader):
                images, labels = trainer.parse_batch_test(batch)
                # image features（共享）
                image_features = model.image_encoder(images.type(model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # logits with robust prompt
                model.load_state_dict({'prompt_learner.ctx': robust_ctx}, strict=False)
                prompts = model.prompt_learner()
                tokenized_prompts = model.tokenized_prompts
                tf0 = model.text_encoder(prompts, tokenized_prompts)
                tf0 = tf0 / tf0.norm(dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits0 = logit_scale * image_features @ tf0.t()

                # logits with personalized prompt
                model.load_state_dict({'prompt_learner.ctx': personal_ctx}, strict=False)
                prompts = model.prompt_learner()
                tokenized_prompts = model.tokenized_prompts
                tf1 = model.text_encoder(prompts, tokenized_prompts)
                tf1 = tf1 / tf1.norm(dim=-1, keepdim=True)
                logits1 = logit_scale * image_features @ tf1.t()

                # 基于置信度的决策级选择
                probs0 = torch.softmax(logits0, dim=1)
                probs1 = torch.softmax(logits1, dim=1)
                preds0 = logits0.argmax(dim=1)
                preds1 = logits1.argmax(dim=1)
                conf0 = probs0.gather(1, preds0.unsqueeze(1)).squeeze(1)
                conf1 = probs1.gather(1, preds1.unsqueeze(1)).squeeze(1)

                equal_mask = preds0 == preds1
                conflict_mask = ~equal_mask
                choose0_mask = conf0 > conf1

                final_logits = torch.empty_like(logits0)
                # 两者一致：直接沿用（任取其一，类别相同）
                if equal_mask.any():
                    final_logits[equal_mask] = logits0[equal_mask]
                # 冲突：选择置信度更高的一方
                if (conflict_mask & choose0_mask).any():
                    final_logits[conflict_mask & choose0_mask] = logits0[conflict_mask & choose0_mask]
                if (conflict_mask & (~choose0_mask)).any():
                    final_logits[conflict_mask & (~choose0_mask)] = logits1[conflict_mask & (~choose0_mask)]

                trainer.evaluator.process(final_logits, labels)
                pred = final_logits.argmax(dim=1)
                for l, p in zip(labels, pred):
                    l_item = l.item()
                    class_total[l_item] += 1
                    if l_item == p.item():
                        class_correct[l_item] += 1

            results = trainer.evaluator.evaluate()
            class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100 for cls in class_correct.keys()}
            return list(results.values()) + [class_accuracy, class_correct, class_total]
    finally:
        # 还原 ctx
        model.load_state_dict({'prompt_learner.ctx': orig_ctx}, strict=False)


def logits_fuse(zero_logits, logits, normalize='mean'):
    # normalize logits
    softmax_fun = torch.nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logits = softmax_fun(zero_logits)
    elif normalize == 'linear':
        zero_logits = zero_logits / torch.norm(zero_logits, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logits, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logits, dim=1, keepdim=True)
        zero_logits = (zero_logits - logits_mean) / logits_std
    else:
        raise RuntimeError("Unsupported normalize mode")

    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize == 'linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise RuntimeError("Unsupported normalize mode")
        current_similarity = current_normalize_logits * zero_logits
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits


def fused_local_fusion_test(mode,
                            trainer_pf,
                            idx,
                            robust_ctx,
                            personal_ctx,
                            beta=0.7,
                            rho=0.25,
                            trainer_pg=None,
                            alpha=0.5,
                            normalize='mean'):
    """
    统一的本地融合评估：mode in ['confidence', 'amu', 'cafo']

    - confidence: 基于置信度的决策级选择（需要 trainer_pf, robust_ctx, personal_ctx）
    - amu: AMU 风格融合（需要 trainer_pf, robust_ctx, personal_ctx, beta, rho）
    - cafo: CaFo 融合（需要 trainer_pf, trainer_pg, robust_ctx, personal_ctx, alpha, normalize）
    返回：list(results.values()) + [class_accuracy, class_correct, class_total]
    """
    mode = (mode or 'confidence').lower()

    if mode == 'confidence':
        return fused_local_test_by_confidence(trainer_pf, idx, robust_ctx, personal_ctx)

    if mode == 'amu':
        model = trainer_pf.model
        sd = model.state_dict()
        orig_ctx = sd['prompt_learner.ctx']
        try:
            trainer_pf.set_model_mode("eval")
            trainer_pf.evaluator.reset()
            model.eval()
            with torch.no_grad():
                data_loader = trainer_pf.fed_test_loader_x_dict[idx]
                from collections import defaultdict
                class_correct = defaultdict(int)
                class_total = defaultdict(int)
                for batch_idx, batch in enumerate(data_loader):
                    images, labels = trainer_pf.parse_batch_test(batch)
                    # image features
                    image_features = model.image_encoder(images.type(model.dtype))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    # logits with robust prompt (GradPur)
                    model.load_state_dict({'prompt_learner.ctx': robust_ctx}, strict=False)
                    prompts = model.prompt_learner()
                    tokenized_prompts = model.tokenized_prompts
                    tf0 = model.text_encoder(prompts, tokenized_prompts)
                    tf0 = tf0 / tf0.norm(dim=-1, keepdim=True)
                    logit_scale = model.logit_scale.exp()
                    logits0 = logit_scale * image_features @ tf0.t()

                    # logits with personalized prompt (PromptFL)
                    model.load_state_dict({'prompt_learner.ctx': personal_ctx}, strict=False)
                    prompts = model.prompt_learner()
                    tokenized_prompts = model.tokenized_prompts
                    tf1 = model.text_encoder(prompts, tokenized_prompts)
                    tf1 = tf1 / tf1.norm(dim=-1, keepdim=True)
                    logits1 = logit_scale * image_features @ tf1.t()

                    # AMU-style fusion: s_fused = s0 + beta * kappa * (s1 - s0)
                    mu = torch.mean(logits0, dim=1, keepdim=True)
                    sigma = torch.std(logits0, dim=1, keepdim=True)
                    eps = torch.tensor(1e-6, device=logits0.device, dtype=logits0.dtype)
                    standardized = (logits0 - mu) / (sigma + eps)
                    moment4 = torch.mean(standardized ** 4, dim=1, keepdim=True)
                    kappa = moment4 ** float(rho)
                    final_logits = logits0 + float(beta) * kappa * (logits1 - logits0)

                    trainer_pf.evaluator.process(final_logits, labels)
                    pred = final_logits.argmax(dim=1)
                    for l, p in zip(labels, pred):
                        l_item = l.item()
                        class_total[l_item] += 1
                        if l_item == p.item():
                            class_correct[l_item] += 1
                results = trainer_pf.evaluator.evaluate()
                class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100 for cls in class_correct.keys()}
                return list(results.values()) + [class_accuracy, class_correct, class_total]
        finally:
            model.load_state_dict({'prompt_learner.ctx': orig_ctx}, strict=False)

    if mode == 'cafo':
        if trainer_pg is None:
            raise ValueError("trainer_pg is required for CaFo mode")
        model_pf = trainer_pf.model
        model_pg = trainer_pg.model
        sd_pf = model_pf.state_dict()
        sd_pg = model_pg.state_dict()
        orig_ctx_pf = sd_pf['prompt_learner.ctx']
        orig_ctx_pg = sd_pg['prompt_learner.ctx']
        try:
            trainer_pf.set_model_mode("eval")
            trainer_pg.set_model_mode("eval")
            trainer_pf.evaluator.reset()
            model_pf.eval(); model_pg.eval()

            model_pf.load_state_dict({'prompt_learner.ctx': personal_ctx}, strict=False)
            model_pg.load_state_dict({'prompt_learner.ctx': robust_ctx}, strict=False)

            with torch.no_grad():
                data_loader = trainer_pf.fed_test_loader_x_dict[idx]
                from collections import defaultdict
                class_correct = defaultdict(int)
                class_total = defaultdict(int)
                for batch_idx, batch in enumerate(data_loader):
                    images, labels = trainer_pf.parse_batch_test(batch)

                    # 三路 logits
                    zs_clip_logits = trainer_pg.zs_clip(images)
                    GradPur_logits = model_pg(images)
                    promptfl_logits = model_pf(images)

                    fuse_logits = logits_fuse(zs_clip_logits, [GradPur_logits, promptfl_logits], normalize=normalize)
                    final_logits = zs_clip_logits + fuse_logits * float(alpha)

                    trainer_pf.evaluator.process(final_logits, labels)
                    pred = final_logits.argmax(dim=1)
                    for l, p in zip(labels, pred):
                        l_item = l.item()
                        class_total[l_item] += 1
                        if l_item == p.item():
                            class_correct[l_item] += 1
                results = trainer_pf.evaluator.evaluate()
                class_accuracy = {cls: (class_correct[cls] / max(class_total[cls], 1)) * 100 for cls in class_correct.keys()}
                return list(results.values()) + [class_accuracy, class_correct, class_total]
        finally:
            model_pf.load_state_dict({'prompt_learner.ctx': orig_ctx_pf}, strict=False)
            model_pg.load_state_dict({'prompt_learner.ctx': orig_ctx_pg}, strict=False)

    if mode == 'wise':
        # WiSE-style parameter fusion for prompt tuning: fuse only prompt_learner.ctx
        model = trainer_pf.model
        sd = model.state_dict()
        orig_ctx = sd['prompt_learner.ctx']
        try:
            trainer_pf.set_model_mode("eval")
            model.eval()
            wise_fracs = [round(x * 0.1, 1) for x in range(0, 11)]
            best_acc = -1.0
            best_frac = 0.0
            best_results_values = None
            best_class_correct = None
            best_class_total = None

            with torch.no_grad():
                data_loader = trainer_pf.fed_test_loader_x_dict[idx]
                for frac in wise_fracs:
                    trainer_pf.evaluator.reset()
                    from collections import defaultdict
                    class_correct = defaultdict(int)
                    class_total = defaultdict(int)

                    fused_ctx = (1.0 - float(frac)) * robust_ctx + float(frac) * personal_ctx
                    # align device/dtype and load into the live parameter
                    fused_ctx_aligned = fused_ctx.to(model.prompt_learner.ctx.device).to(dtype=model.prompt_learner.ctx.dtype)
                    model.load_state_dict({'prompt_learner.ctx': fused_ctx_aligned}, strict=False)

                    for batch_idx, batch in enumerate(data_loader):
                        images, labels = trainer_pf.parse_batch_test(batch)
                        logits = model(images)
                        trainer_pf.evaluator.process(logits, labels)
                        pred = logits.argmax(dim=1)
                        for l, p in zip(labels, pred):
                            l_item = l.item()
                            class_total[l_item] += 1
                            if l_item == p.item():
                                class_correct[l_item] += 1

                    results = trainer_pf.evaluator.evaluate()
                    acc_val = float(list(results.values())[0]) if isinstance(results, dict) and len(results) > 0 else 0.0
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_frac = float(frac)
                        best_results_values = list(results.values())
                        best_class_correct = class_correct
                        best_class_total = class_total

            print(f"client {idx} WiSE (prompt) best frac: {best_frac:.2f}, best acc: {best_acc:.2f}")
            class_accuracy = {cls: (best_class_correct[cls] / max(best_class_total[cls], 1)) * 100 for cls in best_class_correct.keys()}
            return list(best_results_values) + [class_accuracy, best_class_correct, best_class_total]
        finally:
            # restore original prompt
            model.load_state_dict({'prompt_learner.ctx': orig_ctx.to(model.prompt_learner.ctx.device).to(dtype=model.prompt_learner.ctx.dtype)}, strict=False)

    raise ValueError(f"Unsupported fusion mode: {mode}")
