import data_loader.data_loader as data_loader
import feature_extractor.feature_extractor as feature_extractor
from tsfresh.feature_extraction.settings import EfficientFCParameters
import pandas as pd

# Just a helper function that displays all columns of a dataframe
def print_all_cols(x):
    pd.set_option("display.max_columns", None)
    print(x)
    pd.reset_option("display.max_columns")

if __name__ == "__main__":

    # Access to data
    mice_data_dir = '/data/' # Insert directory here
    md = data_loader.DataMerger(mice_data_dir)

    # Name Requirements
    # ID: needs to be an integer (no special characters or letters)
    # Date: DDMMYY
    # format: "id"_"treatment"-IG_"date"_"signal".csv

    # Class "Data Merger": loads the data (+ provides access to the file names)
    print(md.dir)
    print(md.mouse_data_file_map)
    print("Mouse IDs", md.ids)
    print("Signals", md.signals)
    print("Treatments", md.treatments)
    print("Dates", md.dates)

    # Class "Data Container": contains all the data for a given file
    print(md.return_file(mouse_id=169, treat="nea", signal_type="brainsignal", date="14_02_20")) # Overview, i.e. learn how many signals for a file

    # Class "Custom Data Frame": traditional dataframe with advanced functionality
    print(md.return_signal(mouse_id=166, treat="nea", signal_type="running", date="22_02_20", signal=1)) # Overview
    print(md.return_signal(mouse_id=165, treat="nea", signal_type="brainsignal", date="14_02_20", signal=2).df) # Access to dataframe

    # Advanced functionality
    print(md.return_signal(mouse_id=165,
                           treat="nea",
                           signal_type="brainsignal",
                           date="14_02_20", signal=1).left_slice(slice_max=100))

    print(md.return_signal(mouse_id=165,
                           treat="nea",
                           signal_type="brainsignal",
                           date="14_02_20", signal=1).partition_data(chunk_length=50, overlap_ratio=0, remove_shorter=True))


    # Feature Extraction
    test = feature_extractor.FeatureExtractor(md, signal_types=["running", "brainsignal"],
                                                           mice_ids=[166, 165], slice_min=10, target="Nea_vs_other",
                                                           chunk_length=10, slice_max =110, overlap_ratio = 0, dates="all")

    print(test.collected_ts)

    ## First Option: Specify the parameters yourself. Find the overview which features are possible inside the getFeatures function
    fc_parameters = {
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

    # The extracted features can be different for each signal type
    params = {"running": fc_parameters, "brainsignal": fc_parameters}
    print(test.check_dataset())

    feature_matrices = test.getFeatures(feature_dict=params)

    # Gives you to opportunity to check if those features are significantly different between the groups
    print_all_cols(test.getRelevance(feature_matrices, only_significant=True)["running"])

    ## Second option: Extract only significantly different features of a time series (given a predefined set of variables)
    test2 = feature_extractor.FeatureExtractor(md, signal_types=["running", "brainsignal"],
                                              mice_ids=[166, 165], slice_min=10, target="Nea_vs_other",
                                              chunk_length=10, slice_max=110, overlap_ratio=0, dates="all")

    fc_parameters = EfficientFCParameters() # One option of comprehensive set of variables to extract from tsfresh
    # Other parameter options can be found here: https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html (Dec 2020)

    feature_matrices_2, feature_names_2 = test2.relevantFeatures(feature_dict=fc_parameters)

    # Here the relevance table is not calculated, since by definition all features should be significantly different
    # In case, the user needs access to the p-values, it can of course be done and for convenience, the argument
    # "only_significant" argument of the getRelevance function should be set to True.

    print(feature_matrices_2)
    print(feature_names_2)








