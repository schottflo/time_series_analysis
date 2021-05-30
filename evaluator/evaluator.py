import re
from functools import partial
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

class ModelEvaluator:

    """Object that facilitates merging and evaluating the predictions of one or more classifiers"""

    def __init__(self, pred_table_per_signal, target, md):

        self.pred_table_per_signal = pred_table_per_signal

        if target.lower() == "all_vs_all":
            self.labels = np.array(list(md.treatments)) # equivalent to self.treatments in Feature Extraction

        else:
            elements = target.lower().split(sep="_")
            self.labels = np.array([elements[0], elements[2]]) # equivalent to self.treatments in Feature Extraction

        # no extra case for "other" needed here, because it is already taken care of in FeatureExtraction object


    def marginalize_out_signal(self, input, original=True): #leave_out_hemisphere):
        """
        Create a new id without the specific signal based on the input. The input could either be original, i.e. from
        the Classifier class or it could be preprocessed.

        :param input: str
        :param original: boolean
        :return: str
        """

        if original:
            mouse, file, signal, chunk, treat = re.split("-|_", input)
            return (mouse, file, chunk, treat)

        else:
            mouse, date, signal, treat = input
            return (mouse, date, treat)

    def marginalize_out_chunk(self, input, original=True):
        """
        Create a new id without the chunk based on the input. The input could either be original, i.e. from
        the Classifier class or it could be preprocessed.

        :param input: pd.DataFrame
        :param original: list of str and float
        :return:
        """
        if original:
            mouse, file, signal, chunk, treat = re.split("-|_", input)
            return (mouse, file, signal, treat)

        else:
            mouse, date, chunk, treat = input
            return (mouse, date, treat)

    def conduct_single_majority_vote(self, input_row):
        """
        Takes an input row and applies majority voting to it. The predictions are supplied as a list, as well as the
        corresponding prediction confidences. Ties are broken with the average confidence in a class prediction.
        The final prediction is outputted as a str along with the associated average confidence.


        :param input_row: pd.DataFrame
        :return: list of str and float
        """

        true_label = input_row["id"][len(input_row["id"])-1]

        avg_conf = np.zeros(shape=len(self.labels))
        for i in range(len(self.labels)):
            label = self.labels[i]
            if label not in np.array(input_row["predicted_y"]): # skip the given label if it is not mentioned in that row
                continue
            ind = np.where(np.array(input_row["predicted_y"]) == label)
            avg_conf[i] = np.nanmean(np.array(input_row["predicted_y_confidence"])[ind])
            # if np.isnan(avg_conf_label): # skip if
            #     continue
            # else:
             #= avg_conf_label  # average confidence in a class

        predicted_label = self.labels[np.argmax(avg_conf)]

        if true_label == predicted_label:
            return [true_label, np.amax(avg_conf)]
        else:
            return [predicted_label, np.amax(avg_conf)]

    def majority_vote(self, data, first_layer=True, by_chunk=True):
        """
        Wrapper function of conduct_single_majority_vote manipulating the whole dataset. If first_layer, the data is
        coming from the Classifier class directly. Otherwise, it is already preprocessed. If by_chunk, we apply the
        majority voting by chunk, if by_chunk is False, we apply it by signal.

        :param data: pd.DataFrame
        :param first_layer: boolean
        :param by_chunk: boolean
        :return: pd.DataFrame
        """

        if by_chunk:
            grouping_func = self.marginalize_out_chunk
        else:
            grouping_func = self.marginalize_out_signal

        if first_layer:
            original = True
        else:
            original = False

        if original:
            data["id"] = data.index.map(partial(grouping_func, original=original))

        if not original:
            data["id"] = data["id"].map(partial(grouping_func, original=original))

        data = data.groupby('id').agg(list).reset_index()
        data["pooled"] = data.apply(self.conduct_single_majority_vote, axis=1)

        data.drop(["predicted_y", "predicted_y_confidence"], axis=1, inplace=True)
        data[["predicted_y", "predicted_y_confidence"]] = pd.DataFrame(data.pooled.to_list())  # split the pooled column
        data.drop(["pooled"], axis=1, inplace=True)

        return data

    def apply_majority_voting(self, by_chunk=True):
        """
        Preprocess the prediction tables by signal by applying two levels of majority voting and merging them. Majority
        voting can either be applied on the chunk level first (specify by_chunk_first as True) or on the signal level
        first (specify by_chunk_first as False). The result is saved in the object and is not returned.
        
        :param by_chunk: boolean
        :return: None
        """

        predictions_per_signal = {}

        for signal in self.pred_table_per_signal:

            data = self.pred_table_per_signal[signal].drop(["real_y"], axis=1)

            try:
                data.drop(["file_name"], axis=1, inplace=True)
            except:
                pass

            if by_chunk:
                # First level of majority voting
                grouped_data = self.majority_vote(data, first_layer=True, by_chunk=True)

            else:
                grouped_data = self.majority_vote(data, first_layer=True, by_chunk=False)

            grouped_data.rename(columns={"predicted_y": str(signal) + "_pred_y",
                                       "predicted_y_confidence": str(signal) + "_pred_conf"}, inplace=True)

            grouped_data["real_y"] = grouped_data["id"].map(
                lambda x: "other" if "other" in x[len(x) - 1] else x[len(x) - 1])

            predictions_per_signal[signal] = grouped_data

        predictions_list = list(predictions_per_signal.values())

        try:
            predictions = pd.concat(predictions_list, axis=1)
            signal_predictions = predictions.loc[:, ~predictions.columns.duplicated()]

            self.signal_predictions = signal_predictions

            if (signal_predictions.isna().sum().sum() > 0):
                warnings.warn("Some of the test mice do not have data for all signals")
                self.signal_predictions = signal_predictions.dropna()

        except TypeError:
            self.signal_predictions = predictions_list

            if len(self.signal_predictions) == 1:
                self.signal_predictions = self.signal_predictions[0]

        # Pull the preds and confs together in list and then apply majority voting
        ind_pred = ["pred_y" in col for col in self.signal_predictions.columns]
        ind_conf = ["pred_conf" in col for col in self.signal_predictions.columns]

        pred_col = self.signal_predictions.columns[ind_pred]
        conf_col = self.signal_predictions.columns[ind_conf]

        final = pd.DataFrame()
        final["id"] = self.signal_predictions["id"]

        final["predicted_y"] = self.signal_predictions[pred_col].values.tolist()
        final["predicted_y_confidence"] = self.signal_predictions[conf_col].values.tolist()

        final["pooled"] = final.apply(self.conduct_single_majority_vote, axis=1)

        final.drop(["predicted_y", "predicted_y_confidence"], axis=1, inplace=True)
        final[["predicted_y", "predicted_y_confidence"]] = pd.DataFrame(
            final.pooled.to_list())  # split the pooled column
        final.drop(["pooled"], axis=1, inplace=True)

        final["real_y"] = final["id"].map(lambda x: "other" if "other" in x[len(x) - 1] else x[len(x) - 1])

        self.final_prediction = final

    def apply_double_majority_voting(self, by_chunk_first=True):
        """
        Preprocess the prediction tables by signal by applying two levels of majority voting and merging them. Majority
        voting can either be applied on the chunk level first (specify by_chunk_first as True) or on the signal level
        first (specify by_chunk_first as False). The result is saved in the object and is not returned.

        :param by_chunk_first: boolean
        :return: None
        """

        predictions_per_signal = {}

        for signal in self.pred_table_per_signal:

            data = self.pred_table_per_signal[signal].drop(["real_y"], axis=1)

            try:
                data.drop(["file_name"], axis=1, inplace=True)
            except:
                pass

            if by_chunk_first:
                # First level of majority voting
                grouped_data = self.majority_vote(data, first_layer=True, by_chunk=True)
                # Second level of majority voting
                final_data = self.majority_vote(grouped_data, first_layer=False, by_chunk=False)

            else:
                grouped_data = self.majority_vote(data, first_layer=True, by_chunk=False)
                final_data = self.majority_vote(grouped_data, first_layer=False, by_chunk=True)

            final_data.rename(columns={"predicted_y": str(signal) + "_pred_y",
                                 "predicted_y_confidence": str(signal) + "_pred_conf"}, inplace=True)

            final_data["real_y"] = final_data["id"].map(lambda x: "other" if "other" in x[len(x)-1] else x[len(x)-1])

            predictions_per_signal[signal] = final_data

        predictions_list = list(predictions_per_signal.values())

        try:
            predictions = pd.concat(predictions_list, axis=1)
            signal_predictions = predictions.loc[:,~predictions.columns.duplicated()]

            self.signal_predictions = signal_predictions

            if (signal_predictions.isna().sum().sum() > 0):
                warnings.warn("Some of the test mice do not have data for all signals, they will not be considered.")
                self.signal_predictions = signal_predictions.dropna()

        except TypeError:
            self.signal_predictions = predictions_list

        # Pull the preds and confs together in list and then apply majority voting
        ind_pred = ["pred_y" in col for col in self.signal_predictions.columns]
        ind_conf = ["pred_conf" in col for col in self.signal_predictions.columns]

        pred_col = self.signal_predictions.columns[ind_pred]
        conf_col = self.signal_predictions.columns[ind_conf]

        final = pd.DataFrame()
        final["id"] = self.signal_predictions["id"]

        final["predicted_y"] = self.signal_predictions[pred_col].values.tolist()
        final["predicted_y_confidence"] = self.signal_predictions[conf_col].values.tolist()

        final["pooled"] = final.apply(self.conduct_single_majority_vote, axis=1)

        final.drop(["predicted_y", "predicted_y_confidence"], axis=1, inplace=True)
        final[["predicted_y", "predicted_y_confidence"]] = pd.DataFrame(final.pooled.to_list())  # split the pooled column
        final.drop(["pooled"], axis=1, inplace=True)

        final["real_y"] = final["id"].map(lambda x: "other" if "other" in x[len(x)-1] else x[len(x)-1])

        self.final_prediction = final

    def evaluate_signal_classifiers(self, extended_metrics=False):
        """
        Return a set of classification metrics for every classifier (there is one classifier per signal). The output
        is a dictionary mapping the signal_type to a dictionary with the different metrics.

        :param extended_metrics: boolean
        :return: dict of dict
        """

        evaluation = {}

        y_true = self.signal_predictions["real_y"]

        for i in range(len(self.pred_table_per_signal)):

            signal = list(self.pred_table_per_signal)[i]
            y_pred = self.signal_predictions[str(signal) + "_pred_y"]

            acc = balanced_accuracy_score(y_true, y_pred)

            evaluation[signal] = {"acc": acc}

            if extended_metrics:
                prec = precision_score(y_true, y_pred, zero_division="warn", average="weighted")
                rec = recall_score(y_true, y_pred, zero_division="warn", average="weighted")

                evaluation[signal] = {"acc": acc, "prec": prec, "rec": rec}

        return evaluation

    def evaluate_final_classifier(self, extended_metrics=False):
        """
        Return a set of classification metrics for the final classifier.

        :param extended_metrics: boolean
        :return: dict mapping strs to floats
        """
        evaluation = {}

        y_true = self.final_prediction["real_y"]
        y_pred = self.final_prediction["predicted_y"]

        evaluation["acc"] = balanced_accuracy_score(y_true, y_pred)

        if extended_metrics:
            evaluation["prec"] = precision_score(y_true, y_pred, zero_division="warn", average="weighted")
            evaluation["rec"] = recall_score(y_true, y_pred, zero_division="warn", average="weighted")

        return evaluation









