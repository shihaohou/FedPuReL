from Dassl.dassl.utils import Registry, check_availability
# from datasets.caltech101 import Caltech101
from datasets.cifar100 import Cifar100
from datasets.cifar10 import Cifar10
from datasets.cifar10_LT import Cifar10_LT
from datasets.cifar100_LT import Cifar100_LT
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.food101 import Food101
from datasets.food101_LT import Food101_LT
from datasets.oxford_pets_LT import OxfordPets_LT
from datasets.dtd import DescribableTextures
from datasets.dtd_LT import DTD_LT
from datasets.aircraft_LT import Aircraft_LT
from datasets.domainnet import DomainNet
from datasets.office import Office
from datasets.fmnist import FashionMNIST
from datasets.fmnist_LT import FashionMNIST_LT
from datasets.imagenet_LT import ImageNet_LT
from datasets.places_LT import Places_LT
from datasets.stanford_cars_LT import StanfordCars_LT
from datasets.stanford_dogs_LT import StanfordDogs_LT

DATASET_REGISTRY = Registry("DATASET")
# DATASET_REGISTRY.register(Caltech101)
DATASET_REGISTRY.register(Cifar100)
DATASET_REGISTRY.register(Cifar10)
DATASET_REGISTRY.register(Cifar10_LT)
DATASET_REGISTRY.register(Cifar100_LT)
DATASET_REGISTRY.register(OxfordFlowers)
DATASET_REGISTRY.register(OxfordPets)
DATASET_REGISTRY.register(Food101)
DATASET_REGISTRY.register(Food101_LT)
DATASET_REGISTRY.register(OxfordPets_LT)
DATASET_REGISTRY.register(DescribableTextures)
DATASET_REGISTRY.register(DTD_LT)
DATASET_REGISTRY.register(Aircraft_LT)
DATASET_REGISTRY.register(DomainNet)
DATASET_REGISTRY.register(Office)
DATASET_REGISTRY.register(FashionMNIST)
DATASET_REGISTRY.register(FashionMNIST_LT)
DATASET_REGISTRY.register(ImageNet_LT)
DATASET_REGISTRY.register(Places_LT)
DATASET_REGISTRY.register(StanfordCars_LT)
DATASET_REGISTRY.register(StanfordDogs_LT)

def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
