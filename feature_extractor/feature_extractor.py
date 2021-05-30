import os
import warnings
import itertools as iter
import pandas as pd
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_selection.selection import calculate_relevance_table

#these are the different parameter sets from tsfresh (see link in example_data_access.py)
#from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters, ComprehensiveFCParameters

N_CPU = os.cpu_count()

class FeatureExtractor:

    """Object that extracts features from a set of time series"""

    def __init__(self, md, target, mice_ids, signal_types, dates, slice_min, slice_max, chunk_length, overlap_ratio):
        """
        :param md: DataMerger object
        :param target: str
        :param mice_ids: "all" (i.e. a single string) or list of str (even for single mouse_id)
        :param signal_types: "all" (i.e. a single string) or list of str (even for single signal_type)
        :param dates: "all" (i.e. a single string) or list of str (even for single date)
        :param slice_min: float
        :param slice_max: float
        :param chunk_length: float
        :param overlap_ratio: float between [0, 1)
        """
        self.mouse_data = md
        self.target = target

        # "all" vs "all" (= individual pairwise comparisons), "sal vs other" (create one big group of others), "sal vs glu" (specific pairwise)
        elements = target.lower().split(sep="_")

        # indicator for hypothesis like "nea_vs_other"
        self.other = False

        if target.lower() == "all_vs_all":
            self.treatments = self.mouse_data.treatments
        elif "other" in target.lower():
            self.treatments = self.mouse_data.treatments
            self.other = True
            self.main_class = elements[0]
        else:
            self.treatments = set([elements[0], elements[2]])

        # Ids
        if mice_ids == "all":
            self.mice_ids = self.mouse_data.ids
        else:
            self.mice_ids = set(mice_ids)

        # Signal types
        if signal_types == "all":
            self.signal_types = self.mouse_data.signals
        else:
            self.signal_types = set(signal_types)

        # Dates
        if dates == "all":
            self.dates = self.mouse_data.dates
        else:
            self.dates = set(dates)

        # Bring data into format that tsfresh needs

        self.collected_ts = {}
        self.collected_labels = {}
        self.data_preparation(slice_min, slice_max, chunk_length, overlap_ratio)

    def data_preparation(self, slice_min, slice_max, chunk_length, overlap_ratio):
        """
        Prepare the data for feature extraction with tsfresh. In particular constructs a dictionary called collected_ts
        which maps signal_type to pd.DataFrames (which later enables the user to extract features for each signal_type).
        Each pd.DataFrame will have the columns "id", "time" and "signal" (which will be the specified signal name, e.g.
        brainsignal). Also, another dictionary called collected_labels is constructed. It maps signal_type to a mapping from
        each row id to a target class. It is later needed to extract significantly different features for each class

        :param slice_min: int
        :param slice_max: int
        :param chunk_length: int
        :param overlap_ratio: float [0, 1)
        :return: None
        """
        for signal_type in self.signal_types:

            # Extract the composite ids of interest in the following form: (int(id), treatment, signal_type, date)
            mouse_map = list(iter.product(self.mice_ids, self.treatments, [signal_type], self.dates))

            target_map = []
            target_y = []

            signal_type_df = pd.DataFrame()

            for j in mouse_map:

                mouse_id, treat, signal_type, date = j

                # Use DataMerger object to extract necessary data
                # data_gen will be a list of CustomDataFrames for the different signals in a file
                data_gen = self.mouse_data.return_file(mouse_id=mouse_id, treat=treat,
                                                       signal_type=signal_type, date=date).df_list

                if data_gen is None:
                    if mouse_id not in self.mouse_data.ids:
                        warnings.warn(
                            "There is no data available for mouse {0}, which might lead to errors. Only data available for"
                            "the following mice: {1}".format(mouse_id, self.mouse_data.ids))
                    continue

                signal_iterator = 0

                for data in data_gen:

                    # Preprocess the data according to the requirements
                    data = data.right_slice(slice_min=slice_min).left_slice(slice_max=slice_max)
                    chunks = data.partition_data(chunk_length=chunk_length, overlap_ratio=overlap_ratio)

                    signal_iterator += 1

                    chunk_iterator = 0
                    for chunk in chunks:

                        chunk = chunk.df

                        # sometimes chunks have length 0 and we need to skip those chunks
                        if not len(chunk):
                            continue

                        chunk_iterator += 1

                        treat_n = treat

                        # in case of "other" hypothesis
                        if self.other and treat != self.main_class:
                            treat_n = "other({0})".format(treat)

                        # id construction: mouseid-experimentdate-signalid-chunkid_treatmentclass

                        current_id = "{0}-{1}-{2}-{3}_{4}".format(mouse_id, date.replace("_", ""), signal_iterator, chunk_iterator, treat_n)

                        chunk.insert(0, 'id', current_id, True) # pandas automatically repeats the current_id string for each observation

                        chunk.columns = chunk.columns.str.replace("signal_.", signal_type, regex=True) # rename the column into the signal name

                        # Just necessary for tsfresh purposes:
                        target_map.append(current_id)

                        if self.other and treat != self.main_class:
                            target_y.append("other")
                        else:
                            target_y.append(treat)

                        # all chunk dfs stacked in rows (i.e. dataframe will have more rows)
                        if not chunk.isnull().any().any():
                            signal_type_df = signal_type_df.append(chunk)
                        else:
                            warnings.warn("Chunk with ID {0} will not be used for feature extraction, because it contains NAs".format(current_id))

                # Append to readily prepared dataframe for tsfresh
                self.collected_ts[signal_type] = signal_type_df.reset_index(drop=True)
                self.collected_labels[signal_type] = pd.Series(data=target_y, index=target_map)

    def check_dataset(self, skip_na_signals=True, description="Data"):
        """
        Checks if the specifications of the FeatureExtractor result in a valid time series dataset for every signal type,
        i.e. the function checks if observations are available in the given interval. Remove_na_signals indicates whether
        data with missing values of a certain signal type should result in removing the given signal from the analysis.
        Description is an optional string describing the dataset to be checked in order to produce more indicative
        error messages.

        :param description: str
        :param  remove_na_signals: boolean
        :return: boolean (indicating if the dataset is suitable for tsfresh or not)
        """

        data_checks = {}
        for signal in self.collected_ts:
            if not len(self.collected_ts[signal]):
                print("The {0} consisting of {1} did not contain any complete chunks for signal {2} in the given time interval".format(
                            description, self.mice_ids, signal))
                data_checks[signal] = False
            else:
                data_checks[signal] = True

        if skip_na_signals and False in data_checks.values():
            to_be_removed = [key for key, value in data_checks.items() if value is False]
            for signal in to_be_removed:
                del self.collected_ts[signal]

        elif not skip_na_signals and False in data_checks.values():
            return False

        else:
            return True


    def relevantFeatures(self, feature_dict, use_parallel=N_CPU):
        """
        Extracts all significantly different features (at level alpha=0.05) from a set of time series for different signal_types
        and output the resulting feature matrix (with observations on chunk level)

        Overview how tsfresh does that internally: https://tsfresh.readthedocs.io/en/latest/text/feature_filtering.html (Dec 2020)

        :param feature_dict: dict specifying the features to be used
        :param use_parallel: int
        :return: dict with a mapping from each signal type to a tuple containing a feature matrix (incl label) as
        pd.DataFrame() and the names of the parameters
        """
        features_per_signal = {}
        params_per_signal = {}

        for signal in self.collected_ts:
            features_filtered_direct = extract_relevant_features(timeseries_container=self.collected_ts[signal],
                                                                 y=self.collected_labels[signal],
                                                                 column_id='id', column_sort='time', column_value=signal,
                                                                 default_fc_parameters=feature_dict,
                                                                 n_jobs=use_parallel, fdr_level=0.03)

            # the false discovery rate can be adjusted with the additional argument: fdr_level = 0.05 (default)
            relevant_fc_parameters = from_columns(features_filtered_direct)
            print('Identified ', len(features_filtered_direct.columns), ' relevant features.')
            features_filtered_direct['target_class'] = self.collected_labels[signal]
            features_filtered_direct.selection_type = 'relevant'

            features_per_signal[signal] = features_filtered_direct
            params_per_signal.update(relevant_fc_parameters) # dictionary update, i.e. add the keys and values

        return (features_per_signal, params_per_signal)

    def getFeatures(self, feature_dict, n_cpu=N_CPU):
        """
        Extract a given set of features from a time series for different signal_types and output a dictionary of feature
        matrices (incl. labels).

        :param feature_dict: Dictionary mapping signal_type to a dictionary of features (see example)
        :param n_cpu: int
        :return: dict with a mapping from each signal type to a feature matrix (incl label) as pd.DataFrame()
        """
        features_per_signal = {}

        for signal in self.collected_ts:

            extracted_features = extract_features(self.collected_ts[signal], column_id='id', column_sort='time',
                                                  column_value=signal, n_jobs=n_cpu, default_fc_parameters=feature_dict[signal])

            extracted_features.selection_type = 'all'
            extracted_features['target_class'] = self.collected_labels[signal]

            features_per_signal[signal] = extracted_features

        return features_per_signal

    def getRelevance(self, dict_of_feature_matrices, only_significant=False):
        """
        Extract for each feature matrix of a given signal type a table specifying the type of each features (binary,
        real or const), its p-value and whether the Benjamini Hochberg procedure rejected the null hypothesis (i.e.
        if the feature is significantly different between the classes. If only_significant is True, the relevant column
        is omitted and only significantly different features are displayed.

        :param dict mapping signal type to feature matrices
        :return: dict mapping signal type to relevance tables
        """

        extracted_relevance_per_signal = {}

        for signal in self.signal_types:

            feature_matrix = dict_of_feature_matrices[signal].drop(["target_class"], axis=1)
            target = dict_of_feature_matrices[signal]["target_class"]

            rel_table = calculate_relevance_table(feature_matrix, target).reset_index(drop=True)

            if only_significant:
                rel_table = rel_table[rel_table["relevant"] == True].drop(["relevant"], axis=1)

            extracted_relevance_per_signal[signal] = rel_table

        return extracted_relevance_per_signal


    def get_current_feature_names(self, dict_of_feature_matrices):
        """
        Returns a list of extracted features for every signal_type.

        :param dict_of_feature_matrices: dict mapping str to pd.DataFrames
        :return: dict mapping str to lists
        """

        feature_names_per_signal = {}

        for signal in self.signal_types:
            feature_matrix = dict_of_feature_matrices[signal].drop(["target_class"], axis=1)
            feature_names_per_signal[signal] = list(feature_matrix.columns)

        return feature_names_per_signal

    def keep_track_of_features(self, feature_dict, current_features):
        """
        Update the global feature dictionary that maps each feature name to the times it was mentioned (usually used
        in leave-one-mouse-out CV, when the user is interested how many times a feature was extracted)

        :param feature_dict: dict mapping str to ints
        :param current_features: dict mapping str to lists
        :return: dict mapping str to ints
        """

        for signal in self.signal_types:

            new_features = {feature: 1 for feature in current_features[signal]}

            if len(feature_dict) == 0:
                feature_dict = new_features
            else:
                for key in new_features:
                    if key in feature_dict:
                        feature_dict[key] += new_features[key]
                    else:
                        feature_dict[key] = new_features[key]

        return feature_dict



