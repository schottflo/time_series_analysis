import data_loader.data_loader as data_loader
import feature_extractor.feature_extractor as feature_extractor
import classifier.classifier as classifier
import evaluator.evaluator as eval

import itertools as it
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tsfresh.feature_extraction.settings import EfficientFCParameters

# Note: Hyperparameter Search should only be conducted for one signal_type at the time
# Reason: Otherwise it will be hard to arrive at clear conclusions (some features might be consistent for "running", while features of "brainsignal")

# Problem-specific parameters
TARGET = "all_vs_all"
SIGNAL_TYPE = ["brainsignal"] # only one signal recommended at a time, takes long anyway

# Hypothesis-dependent parameters (i.e. TARGET variable)
SAL_MICE = {165, 166, 176, 218, 306, 307, 413}
NEA_MICE = {165, 166, 167, 168, 176, 299, 302, 303, 306, 307, 327, 413}
MICE = SAL_MICE.union(NEA_MICE)

# Parameter grid
STARTS = [40]
ENDS = [90, 100, 60] # 3 values
CHUNKS = list(range(10, 60, 10))  # 5 values
OVERLAPS = list(range(0, 100, 10)) # in % # 10 values

# Machine-specific parameter
DIR = '/data/' # directory where the files are saved

# Just a helper function that prints all columns of a dataframe
def print_all_cols(x):
    pd.set_option("display.max_columns", None)
    print(x)
    pd.reset_option("display.max_columns")

def hyperparameter_search():

    # Load the data
    md = data_loader.DataMerger(DIR)

    # Number of parameters to be searched
    num_param = len(STARTS) * len(ENDS) * len(CHUNKS) * len(OVERLAPS)
    print("{0} parameters will be searched.".format(num_param))

    # Dictionary to store the results
    results = {}
    ts_variables = EfficientFCParameters()

    parameters = list(it.product(STARTS, ENDS, CHUNKS, OVERLAPS)) # creates the different combinations

    for combination in parameters:

        start, end, chunk_size, overlap = combination

        print("------------------------------")
        print("Current parameter combination:")
        print("Start time:", start)
        print("End time:", end)
        print("Chunk Length:", chunk_size)
        print("Chunk Overlap:", overlap / 100)
        print("------------------------------")

        if (end - start) < chunk_size:
            results[combination] = [np.nan, "Chunk size too large for time window."]
            continue

        acc_vals = []
        weights = []
        features = {}
        important_features = {}

        for j in MICE:

            training_mice = (MICE - {j})
            test_mouse = {j}

            print("\nTesting mouse", test_mouse, "\n")

            # Initialize training set (i.e. load data of the given mice and slice up like specified)
            train_feature_generator = feature_extractor.FeatureExtractor(md=md, signal_types=SIGNAL_TYPE,
                                                                         mice_ids=training_mice, slice_min=start,
                                                                         target=TARGET, chunk_length=chunk_size,
                                                                         slice_max=end, overlap_ratio=overlap / 100,
                                                                         dates="all")

            # Initialize test set (i.e. load data of the given mice and slice up like specified)
            test_feature_generator = feature_extractor.FeatureExtractor(md=md, signal_types=SIGNAL_TYPE,
                                                                        mice_ids=test_mouse, slice_min=start,
                                                                        target=TARGET, chunk_length=chunk_size,
                                                                        slice_max=end, overlap_ratio=overlap / 100,
                                                                        dates="all")

            # Check if all test data and training data is available in the given interval
            if not test_feature_generator.check_dataset(description="test_{0}".format(test_mouse)) or \
                    not train_feature_generator.check_dataset(description="train_{0}".format(training_mice)):
                acc_score = np.nan
                acc_vals.append(acc_score)
                weights.append(0)
                continue

            # Extract the significantly features from the given training set (could also be done with predefined features -> then specify ts_variables by hand and use getFeatures)
            train_feature_matrices, train_feature_names = train_feature_generator.relevantFeatures(feature_dict=ts_variables)

            # Skip if no features are significantly different
            if len(train_feature_names) == 0:
                acc_score = np.nan
                acc_vals.append(acc_score)
                weights.append(0)
                continue

            # Extract the features of the current training set and add them to global list of extracted features
            current_features = train_feature_generator.get_current_feature_names(train_feature_matrices)
            features = train_feature_generator.keep_track_of_features(feature_dict=features,
                                                                      current_features=current_features)

            # Extract the significantly different features from the training set for the test set
            test_feature_matrices = test_feature_generator.getFeatures(feature_dict=train_feature_names)

            # Merge training and test sets
            feature_matrices = train_feature_matrices
            for signal in train_feature_matrices:
                feature_matrices[signal] = feature_matrices[signal].append(test_feature_matrices[signal])
                feature_matrices[signal].dropna(inplace=True) # In some rare cases, there are NAs which need to be dropped

            # Use the extracted features to compute predictions on the test set
            model = LogisticRegression(random_state=42)#RandomForestClassifier(random_state=42)  # can be changed
            classification_model = classifier.Classifier(data_matrices=feature_matrices, model=model)
            classification_model.train_model(train_test={'train': list(training_mice), 'test': [test_mouse]},
                                             adjust_datasets_to_avail=True)

            # For any classifier, we extract the 10 most important features based on permutation importance (based on training set)
            # Link: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance

            if len(classification_model.pred_table_per_signal) == 0:
                acc_score = np.nan
                acc_vals.append(acc_score)
                weights.append(0)
                continue

            # Extract 10 most important features
            current_important_features = {}
            for signal in SIGNAL_TYPE:
                imp = classification_model.importances[signal]["importances_mean"]
                ind = np.argsort(imp)[::-1]
                current_important_features[signal] = np.array(current_features[signal])[ind[:10]]

            important_features = train_feature_generator.keep_track_of_features(feature_dict=important_features,
                                                                         current_features=current_important_features)

            print(features)
            print(important_features)

            # Return predictions
            predictions = classification_model.return_predictions(add_location=True, md=md)

            print_all_cols(predictions)

            # Compute the weights (i.e. how many samples per testset)
            weight = predictions[SIGNAL_TYPE[0]].shape[0] # this is a "shortcut" - it assumes that we are only evaluating one signal at once, which we should do in hyperparameter search

            # Instantiate evaluator
            model_evaluator = eval.ModelEvaluator(predictions, target=TARGET, md=md)

            # Apply the majority voting
            model_evaluator.apply_majority_voting(by_chunk=True)

            print_all_cols(model_evaluator.signal_predictions)

            result_one_mouse = model_evaluator.evaluate_final_classifier(extended_metrics=True)
            print(result_one_mouse)

            acc_vals.append(result_one_mouse["acc"])
            weights.append(weight)

        avg_acc = np.nansum(np.array(acc_vals)*np.array(weights))/sum(weights) # this average is weighted for the overall number of observations for a given mouse
        print("Avg Accuracy: ", avg_acc)
        print(acc_vals)
        print(important_features)

        results[combination] = {"avg_acc": avg_acc, "acc_vals": acc_vals, "imp_feat": important_features, "all_feat": features}
        np.save('result_backup_{0}.npy'.format(combination), results)

    print(results)
    np.save('result.npy', results)

if __name__=="__main__":
    hyperparameter_search()



