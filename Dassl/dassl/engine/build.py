from Dassl.dassl.utils import Registry, check_availability
from trainers.purelAdapter import FedClip
from trainers.purelGPAdapter import AdapterGradPur
from trainers.purelLora import ClipLora
from trainers.purelLoraGP import LoRAGradPur
from trainers.purelPrompt import PromptFL
from trainers.purelPromptGP import GradPur

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(FedClip)
TRAINER_REGISTRY.register(ClipLora)
TRAINER_REGISTRY.register(GradPur)
TRAINER_REGISTRY.register(LoRAGradPur)
TRAINER_REGISTRY.register(AdapterGradPur)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    # print("avai_trainers",avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
