import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from loralib.utils import (
    apply_lora, get_lora_parameters
)

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'LoRAGradPur',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        pos_embed = self.positional_embedding.to(prompts.device).type(self.dtype)
        x = prompts + pos_embed
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = "a photo of a"
        ctx_init = ctx_init.replace("_", " ")

        prompt_prefix = ctx_init
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer('tokenized_prompts', tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer('embedding', embedding)

        self.n_cls = n_cls
        self.name_lens = name_lens
        self.class_token_position = getattr(cfg.TRAINER, 'PROMPTFL', cfg.TRAINER).CLASS_TOKEN_POSITION if hasattr(cfg, 'TRAINER') else 'end'

    def forward(self):
        return self.embedding


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_features=False):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if return_features:
            return image_features, logits, text_features
        else:
            return logits


CUSTOM_TEMPLATES = {
    "OxfordPets": "a type of pet, a photo of a {}.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "CIFAR10": "a photo of a {}.",
    "CIFAR100": "a photo of a {}.",
    "CIFAR100_LT": "a photo of a {}.",
    "CIFAR10_LT": "a photo of a {}.",
    "fmnist_LT": "a photo of a {}.",
    "Cifar100_LT": "a photo of a {}.",
    "Cifar10_LT": "a photo of a {}.",
}


class ZSCLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        text_features = self.text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits


class GradPurLoss(nn.modules.loss._Loss):
    def __init__(self, T):
        super(GradPurLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T, -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()
        return xe_loss, kl_loss


class LoRAParamModule(nn.Module):
    def __init__(self, lora_params):
        super().__init__()
        for idx, p in enumerate(lora_params):
            self.register_parameter(f"param_{idx}", p)

    def forward(self):
        return None


class LoRAGradPur(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        # Apply LoRA to CLIP encoders
        apply_lora(cfg.TRAINER.CLIPLORA, clip_model)

        # Build CustomCLIP (will use LoRA-inserted encoders)
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.to(self.device)

        # Multi-GPU (optional)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1], output_device=0)

        # Freeze non-LoRA params; optimize LoRA only
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

        lora_params = list(get_lora_parameters(self.model))
        self.optim = build_optimizer(lora_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # Register a lightweight module that only exposes LoRA params for GradPur stepping
        self.lora_param_module = LoRAParamModule(lora_params)
        self.register_model("lora_params", self.lora_param_module, self.optim, self.sched)

        # Also register full model for saving/eval bookkeeping (no optimizer/scheduler)
        self.register_model("clip_lora_model", self.model, None, None)

        # Teacher: zero-shot CLIP
        self.zs_clip = ZSCLIP(cfg, classnames).cuda()
        self.cls_num_list = self.get_cls_num_list()
        self.criterion = GradPurLoss(T=1)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

    def get_cls_num_list(self):
        y_train = self.dm.dataset.y_train
        cls_num_list = [0] * self.num_classes
        for label in y_train:
            cls_num_list[label] += 1
        # print("cls_num_list:", cls_num_list)
        return cls_num_list
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.PROMPTFL.PREC if hasattr(self.cfg.TRAINER, 'PROMPTFL') else 'fp32'
        with autocast() if prec == "amp" else torch.cuda.amp.autocast(enabled=False):
            student_logits = self.model(image)

        with torch.no_grad():
            teacher_logits = self.zs_clip(image)

            # 熵计算（不求导）
            def entropy_from_logits(logits):
                logits = logits.float()
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                ent = -(probs * log_probs).sum(dim=-1)
                return ent


            def search_temperature_per_sample(logits, target_entropy, iters=20, lo=0.05, hi=20.0):
                B = logits.shape[0]
                lo_t = torch.full((B,), lo, device=logits.device, dtype=torch.float32)
                hi_t = torch.full((B,), hi, device=logits.device, dtype=torch.float32)

                def entropy_at_tau(tau_vec):
                    tau = tau_vec.view(-1, 1)
                    return entropy_from_logits(logits / tau)

                for _ in range(iters):
                    mid = (lo_t + hi_t) * 0.5
                    ent_mid = entropy_at_tau(mid)
                    go_left = ent_mid > target_entropy
                    hi_t = torch.where(go_left, mid, hi_t)
                    lo_t = torch.where(go_left, lo_t, mid)

                return (lo_t + hi_t) * 0.5


            ent_t = entropy_from_logits(teacher_logits)
            ent_s = entropy_from_logits(student_logits)
            h_star = 0.5 * (ent_t + ent_s)


            tau_t = search_temperature_per_sample(teacher_logits, h_star)
            tau_s = search_temperature_per_sample(student_logits, h_star)


            p_t = F.softmax(teacher_logits / tau_t.view(-1, 1), dim=-1)


        p_s = F.softmax(student_logits / tau_s.view(-1, 1), dim=-1)


        kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - torch.log(p_s.clamp_min(1e-12)))).sum(dim=-1)
        kl_loss = kl.mean()


        xe_loss = F.cross_entropy(student_logits, label)


        self.GradPur_backward_and_update(xe_loss, kl_loss, self.cfg.LOSS.LAMBDA, names=["lora_params"]) 

        loss_summary = {
            "xe_loss": xe_loss.item(),
            "kl_loss": kl_loss.item(),
            "acc": compute_accuracy(student_logits, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def lora_state_dict(self):
        my_state_dict = self.model.state_dict()
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}

    def load_lora_state_dict(self, state_dict):
        current_state = self.model.state_dict()
        for key in state_dict:
            if key in current_state:
                current_state[key].copy_(state_dict[key])
        self.model.load_state_dict(current_state, strict=False)


