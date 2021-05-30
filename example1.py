import data_loader.data_loader as data_loader
import feature_extractor.feature_extractor as feature_extractor
import classifier.classifier as classifier
import evaluator.evaluator as eval

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Just a helper function to show all columns of a dataframe
def print_all_columns(x):
    pd.set_option("display.max_columns", None)
    print(x)
    pd.reset_option("display.max_columns")

# 1. Use case: Extracting a set of features from a time series and classifying based on it

# Problem-specific parameters
TARGET = "sal_vs_nea"
SIGNAL_TYPES = ["brainsignal", "running"]
MICE = [414, 296, 176] # 166, , 303, 327, 165
START = 10
END = 100
CHUNK_LEN = 20
OVERLAP = 0

# Machine-specific parameter
DIR = 'C:/Users/Flori/Documents/Jobs/ETH Neurobehavioral Dynamics Lab/future_data' # directory where the files are saved

def base_classification():

    md = data_loader.DataMerger(DIR)
    data_extractor = feature_extractor.FeatureExtractor(md=md,  target=TARGET, signal_types=SIGNAL_TYPES, mice_ids=MICE,
                                                        slice_min=START, chunk_length=CHUNK_LEN, slice_max=END,
                                                        overlap_ratio=OVERLAP, dates="all")

    feature_names = {
        "large_standard_deviation": [{"r": 0.05}, {"r": 0.1}],
        "abs_energy": None,
        "autocorrelation": [{"lag": 3}],
        "variance": None,
        "number_peaks": [{"n": 10}],
        "count_above_mean": None,
        "longest_strike_below_mean": None,
        "mean": None,
        "maximum": None,
        "median": None,
        "variance": None
    }

    feature_dict = {"brainsignal": feature_names, "running": feature_names}

    feature_matrices = data_extractor.getFeatures(feature_dict=feature_dict)

    model = RandomForestClassifier(random_state=42) # can be changed
    classification_model = classifier.Classifier(data_matrices=feature_matrices, model=model)

    classification_model.train_model(train_test={'train': [165, 166, 414, 303, 327], 'test': [176, 296]},
                                     adjust_datasets_to_avail=True)

    predictions = classification_model.return_predictions(add_location=True, md=md)

    print(predictions)

    model_evaluator = eval.ModelEvaluator(predictions, target=TARGET, md=md)

    model_evaluator.apply_double_majority_voting(by_chunk_first=True) # no return value!

    print_all_columns(model_evaluator.signal_predictions) # Predictions for mouse 296 available for brainsignal

    # Signal level analysis
    print(model_evaluator.evaluate_signal_classifiers(extended_metrics=True))

    # Analysis of final predictions
    print(model_evaluator.evaluate_final_classifier(extended_metrics=True))

if __name__=="__main__":
    base_classification()


