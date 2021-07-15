import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .custom_dataset import CustomDataset

dataset_factory = {
	'custom': CustomDataset
}

def get_dataset(dataset):
	return dataset_factory[dataset]
