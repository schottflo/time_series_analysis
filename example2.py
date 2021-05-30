import data_loader.data_loader as data_loader
import feature_extractor.feature_extractor as feature_extractor
import classifier.classifier as classifier
import evaluator.evaluator as evaluator

import pandas as pd

from tsfresh.feature_extraction.settings import EfficientFCParameters

# 2. Use case: Find significantly different features of different time series

# Problem-specific parameters
TARGET = "sal_vs_other" # "other" has to be named last
SIGNAL_TYPES = ["brainsignal", "running"]
MICE = [414, 166, 165, 303]
START = 10
END = 100
CHUNK_LEN = 20
OVERLAP = 0.4

# Machine-specific parameter
DIR = '/data/' # directory where the files are saved

def feature_extraction():

    md = data_loader.DataMerger(DIR)
    data_extractor = feature_extractor.FeatureExtractor(md=md, target=TARGET, signal_types=SIGNAL_TYPES, mice_ids=MICE,
                                                        slice_min=START, chunk_length=CHUNK_LEN, slice_max=END,
                                                        overlap_ratio=OVERLAP, dates="all")

    parameters = EfficientFCParameters()

    feature_matrices, feature_names = data_extractor.relevantFeatures(feature_dict=parameters)
    print(feature_matrices)

    print(data_extractor.getRelevance(feature_matrices, only_significant=False)["brainsignal"]) # because running has no significantly different features here

if __name__=="__main__":
    feature_extraction()
