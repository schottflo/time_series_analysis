import warnings
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def print_all_cols(x):
    pd.set_option("display.max_columns", None)
    print(x)
    pd.reset_option("display.max_columns")

class Classifier:

    """ Object that prepares training and testing set, fits the classification model and produces the predictions"""

    def __init__(self, data_matrices, model):
        """
        Instantiate Classifier object.

        :param data_matrices: dict mapping str (signal_type) to pd.DataFrame (data_matrix for a given signal_type)
        :param model: sklearn classifier (i.e. needs to have fit(), predict(), predict_proba() method implemented)
        """

        self.data_matrices = data_matrices

        self.model = {}

        for signal in data_matrices:
            self.model[signal] = model

        self.importances = {}

    def train_model(self, train_test, adjust_datasets_to_avail=True):
        """
        Train one model per feature and output the corresponding prediction tables, which specify for each chunk
        the real y, the predicted y and the probability of the class given the model (which is minimal at 0.5 and
        maximal at 1 for binary classification). The argument train_test lets the user specify which mice are used
        for training and which for testing. Adjust_datasets_to_avail takes care of the case when certain signals are
        not available in the training or test set.

        :param train_test: dict mapping "train" or "test" to a lists of mice ids
        :param adjust_datasets_to_avail: boolean
        :return: dict mapping signal_type to pred_tables
        """
        pred_table_per_signal = {}

        for signal in self.data_matrices:

            X = self.data_matrices[signal].drop(columns='target_class', axis=1)
            y = self.data_matrices[signal]['target_class']

            subjects = train_test['train']

            train_index = [any([str(subject) in i for subject in subjects]) for i in list(self.data_matrices[signal].index)]
            test_index = [not i for i in train_index]

            X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]

            if adjust_datasets_to_avail:

                if X_test.shape[0] == 0:
                    warnings.warn("There is no test data for {0} available".format(signal))
                    continue

                if X_train.shape[0] == 0:
                    warnings.warn("There is no training data for {0} available".format(signal))
                    continue

            # Re-scaling the features, can help the optimization algorithm to converge
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Handle class imbalance by assigning weights to the samples in the loss function
            sample_weights = compute_sample_weight('balanced', y_train)

            self.model[signal].fit(X=X_train, y=y_train, sample_weight=sample_weights)

            self.importances[signal] = permutation_importance(estimator=self.model[signal], X=X_train, y=y_train,
                                                             scoring="balanced_accuracy", random_state=42)


            pred_table = self.construct_pred_table(X_test, y_test, signal)

            pred_table_per_signal[signal] = pred_table

        self.pred_table_per_signal = pred_table_per_signal

    def construct_pred_table(self, X_test, y_test, signal):
        """
        Construct a prediction table of a given signal based on a test set. A prediction table specifies for each chunk
        the real y, the predicted y and the prediction probability. Also, output the feature importance on the test set.

        :param X_test: pd.DataFrame
        :param y_test: pd.Series
        :param signal: str
        :return: pd.DataFrame
        """
        y_out = self.model[signal].predict(X_test)
        y_out_confidence = np.apply_along_axis(np.max, 1, self.model[signal].predict_proba(X_test))

        out_df = pd.DataFrame({'real_y': y_test, 'predicted_y': y_out, 'predicted_y_confidence': y_out_confidence})

        return out_df

    def return_predictions(self, add_location=False, md=False):
        """
        Add column to pred_tables to specify location of the data that were classified (if add_location) and then return
        the prediction tables.

        :param add_location: boolean
        :param md: DataMerger object
        :return: dict mapping str to pd.DataFrame
        """

        if add_location:
            for signal in self.pred_table_per_signal:

                file_names = []
                for id in self.pred_table_per_signal[signal].index:


                    id_info = id.split("-")

                    mouse_id = int(id_info[0])
                    treat = id_info[len(id_info)-1].split("_")[1]

                    if "other" in treat:
                        treat = treat.split("(")[1][:(len(treat.split("(")[1])-1)]

                    date = id_info[1][:2] + "_" + id_info[1][2:4] + "_" + id_info[1][4:]

                    file_name = md.mouse_data_file_map[(mouse_id, treat, signal, date)].split("/").pop() # (int(id), treatment, signal_type, date)
                    file_names.append(file_name)

                self.pred_table_per_signal[signal]["file_name"] = file_names

        return self.pred_table_per_signal