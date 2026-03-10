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
from torch.nn.modules.loss import _Loss

# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid

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
    design_details = {"trainer": 'PromptFL',
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
        x = prompts + self.positional_embedding.type(self.dtype)
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


class CLIP(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1,
                                                               keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()

        text_features = self.text_features
        text_features = text_features.to(image_features.device)
        logits = logit_scale * image_features @ text_features.t()
        return logits



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype



        self.img_adap = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim),
            nn.Tanh(),
            nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim),
            nn.Softmax(dim=1)
        ).to(self.dtype)



    def forward(self, image, return_features=False):



        image_features = self.image_encoder(image.type(self.dtype))
        image_features_att = self.img_adap(image_features)
        image_features = torch.mul(image_features_att, image_features)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if return_features:
            return image_features, logits, text_features
        else:
            return logits


class GradPurLoss(_Loss):
    def __init__(self, T):
        super(GradPurLoss, self).__init__()
        self.T = T

    def forward(self, stu_logits, tea_logits, label):
        xe_loss = F.cross_entropy(stu_logits, label)

        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss

# @TRAINER_REGISTRY.register("PromptFL")
class AdapterGradPur(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL.PREC == "fp32" or cfg.TRAINER.PROMPTFL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Building zeroshot CLIP")
        self.zs_clip = CLIP(cfg, classnames)

        print("Turning off gradients in ZS Clip model")
        for name, param in self.zs_clip.named_parameters():
            param.requires_grad_(False)


        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "img_adap" not in name:
                param.requires_grad_(False)
        for param in self.model.img_adap.parameters():
            param.requires_grad = True

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        self.cls_num_list = self.get_cls_num_list()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.zs_clip.to(self.device)

        self.optim = build_optimizer(self.model.img_adap.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("img_adap", self.model.img_adap, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL.PREC == "amp" else None



        self.criterion = GradPurLoss(T=1)

    def forward_backward(self, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL.PREC

        # output = self.model(image)
        _, output, _ = self.model(image, return_features=True)
        with torch.no_grad():
            zs_clip_output = self.zs_clip(image)

        xe_loss, kl_loss = self.criterion(output,
                                            zs_clip_output.detach(),
                                            label)
        self.GradPur_backward_and_update(xe_loss, kl_loss,
                                                self.cfg.LOSS.LAMBDA)

        loss_summary = {
            "xe_loss": xe_loss.item(),
            "kl_loss": kl_loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def mask_grad(self, labels):
        unique_labels = torch.unique(labels)
        for name, param in self.model.named_parameters():
            if name == "prompt_learner.general_ctx":
                # Do nothing, allow all gradients to pass
                pass
            elif name == "prompt_learner.class_aware_ctx":
                grad_mask = torch.zeros_like(param.data)
                grad_mask[unique_labels, :, :] = 1
                param.grad.data.mul_(grad_mask)

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def get_cls_num_list(self):
        y_train = self.dm.dataset.y_train
        cls_num_list = [0] * self.num_classes
        for label in y_train:
            cls_num_list[label] += 1
        # print("cls_num_list:", cls_num_list)
        return cls_num_list

    def communication(self, client_models, client_weights):
        with torch.no_grad():
            for key in self.model.img_adap.state_dict().keys():
                temp = torch.zeros_like(self.model.img_adap.state_dict()[key])
                for client_idx, client_model in enumerate(client_models):
                    temp += client_weights[client_idx] * client_model.img_adap.state_dict()[key]
                self.model.img_adap.state_dict()[key].data.copy_(temp)
                for client_model in client_models:
                    client_model.img_adap.state_dict()[key].data.copy_(self.model.img_adap.state_dict()[key])
        return self.model, client_models

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model bash main.sh caltech101 rn50_ep50 end 16 1 Falsenot found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    def collect_gradients(self):

        grad_vectors = []
        

        for name, param in self.model.img_adap.named_parameters():
            if param.grad is not None:
                grad_vectors.append(param.grad.view(-1))

        if grad_vectors:
            grad_vec = torch.cat(grad_vectors).detach().cpu()
            

            if self.gradient_collector['grad_sum'] is None:
                self.gradient_collector['grad_sum'] = grad_vec.clone()
            else:
                self.gradient_collector['grad_sum'] += grad_vec
            
            self.gradient_collector['batch_gradients'].append(grad_vec)
            self.gradient_collector['batch_count'] += 1
    
    def compute_local_avg_gradient(self):

        if self.gradient_collector['batch_count'] > 0:
            self.local_avg_gradient = self.gradient_collector['grad_sum'] / self.gradient_collector['batch_count']
        else:
            self.local_avg_gradient = None

        self.gradient_collector = {
            'batch_gradients': [],
            'batch_count': 0,
            'grad_sum': None,
        }
        
        return self.local_avg_gradient
    
