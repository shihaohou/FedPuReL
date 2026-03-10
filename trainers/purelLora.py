import os.path as osp
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import Registry
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from loralib.utils import (
    apply_lora, mark_only_lora_as_trainable, get_lora_parameters
)

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ClipLora',
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
        # 将 positional_embedding 注册为 buffer
        # self.register_buffer('positional_embedding', clip_model.positional_embedding)
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # x = prompts + self.positional_embedding.type(self.dtype)
        pos_embed = self.positional_embedding.to(prompts.device).type(self.dtype)
        x = prompts + pos_embed

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
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
        print(f'Initial context: "{prompt_prefix}"')
        # 生成 prompts
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
        self.class_token_position = cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION

    def forward(self):
        return self.embedding


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

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


# @TRAINER_REGISTRY.register("PromptFL")
class ClipLora(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print(f"[ClipLora] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        clip_model.float()
        print("[ClipLora] Building CustomCLIP")
        lora_layers = apply_lora(cfg.TRAINER.CLIPLORA, clip_model)
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.to(self.device)
        # 将整个模型移到指定设备（例如 cuda:0）

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1], output_device=0)

        if cfg.TRAINER.CLIPLORA.ENABLED:
            print("[ClipLora] Applying LoRA to CLIP encoders ...")

            for name, param in self.model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad_(False)


            total_lora_params = sum(p.numel() for p in get_lora_parameters(self.model) if p.requires_grad)
            print(f"[ClipLora] # LoRA parameters: {total_lora_params}")

            lora_params = get_lora_parameters(self.model)
            self.optim = build_optimizer(lora_params, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

            self.register_model("clip_lora_model", self.model, self.optim, self.sched)



        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None

        self.cls_num_list = self.get_cls_num_list()

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)

        logits = self.model(image)
        loss = F.cross_entropy(logits, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def get_cls_num_list(self):
        y_train = self.dm.dataset.y_train
        cls_num_list = [0] * self.num_classes
        for label in y_train:
            cls_num_list[label] += 1
        # print("cls_num_list:", cls_num_list)
        return cls_num_list

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




