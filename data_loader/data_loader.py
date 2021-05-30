import os
import warnings
import pandas as pd
import numpy as np

class DataMerger:

    """Object that loads the data into Python and grants the user access to the preprocessed data"""

    def __init__(self, dir):

        if dir is None:
            raise ValueError('Directory path cannot be None')
        if not os.path.isdir(dir):
            raise ValueError('Inexistent directory provided')

        self.dir = dir
        self.mouse_data_file_map = {}

        self.ids = set()
        self.treatments = set()
        self.signals = set()
        self.dates = set()

        # Build mouse_data_file_map
        self.preprocess_dir()

    def preprocess_dir(self):
        """
        Builds a dictionary mapping each (id_treatm_signal) combination to one or more files (more than one if more than one experiment)

        :return: None (only builds self.mouse_data_file_map)
        """

        for file in os.listdir(self.dir): # iterates through the files in a directory alphabetically sorted

            file_lower = file.lower() # make the name lower case

            file_type = file_lower[-3:] # extract file type
            file_name = file_lower[:-4] # extract file name (everything except the .csv)

            if file_type != "csv": # don't look at non-csv files
                continue

            name_elements = file_name.split(sep="_") # extract the elements of the name and process them

            id = name_elements[0]
            treatment = name_elements[1].split(sep="-")[0]
            signal_type = name_elements.pop()

            date_raw = name_elements[2]
            date = date_raw[:2] + "_" + date_raw[2:4] + "_" + date_raw[4:]

            if len(id) == 0 or len(treatment) == 0 or len(signal_type) == 0 or len(date) == 0:
                raise ValueError('File {0} is of wrong name format.'.format(file))

            try:
                mouse_data_id = (int(id), treatment, signal_type, date)
            except ValueError:
                raise ValueError('File {0} is of the wrong name format'.format(file))

            self.ids.add(int(id))
            self.treatments.add(treatment)
            self.signals.add(signal_type)
            self.dates.add(date) # No data object, just a string

            self.mouse_data_file_map[mouse_data_id] = self.dir + "/" + str(file) #[(os.path.join(self.dir, file))]

    def preprocess_df(self, df, id, separate=True):
        """
        Preprocess dataframe by naming columns and separating individual signals within a signal type if there are any

        :param df: pd.DataFrame
        :param id: quadruple in the form (mouse_id, treatment, signal, date) where mouse_id is int and the others are str
        :param separate: boolean to indicate whether to separate individual signals
        """

        col_names = [column for column in df.columns]
        num_col = len(col_names)

        if num_col > 1:
            signal_names = ["signal_{0}".format(i+1) for i in range(num_col-1)]
            df.columns = ["time", *signal_names]
        else:
            return None

        if separate:
            signal_series = [df[col].squeeze() for col in df.columns[1:]]
            df_list = [CustomDataFrame(pd.concat([df["time"].squeeze(), signal], axis=1), id) for signal in signal_series]

            return df_list

        return [CustomDataFrame(df)]


    def return_file(self, mouse_id, treat, signal_type, date):
        """
        Access all data for a specific (id_treatm_signal) combination. Returns a list of list where each file for a
        specific (id_treatm_signal) combination is broken down to one or more signals contained in the supplied csv
        (one or more columns)

        :param mouse_id: int or str
        :param treat: str
        :param signal: str
        :param date: str
        :return: list of lists
        """

        treat, signal_type, date = treat.lower(), signal_type.lower(), date
        mouse_signal_file_id = (int(mouse_id), treat, signal_type, date)

        if mouse_signal_file_id not in self.mouse_data_file_map:
            return DataContainer(df_list=None)
            #raise ValueError("{0} is an invalid combination; check the validity of the id, treatment, signal type and date".format(mouse_signal_file_id))

        file = self.mouse_data_file_map[mouse_signal_file_id]
        return DataContainer(df_list=self.preprocess_df(df=pd.read_csv(file), id=mouse_signal_file_id))

    def return_signal(self, mouse_id, treat, signal_type, date, signal):
        """
        Return a specific dataframe

        :param mouse_id: int or str
        :param treat: str
        :param signal: str
        :param date: str
        :param signal: int (column number in the original file, i.e. when using 3 columns in original file, can be within [1, 3])
        :return: pd.DataFrame
        """

        if signal <= 0:
            raise ValueError("{0} is not a sensible value for signal".format(signal))

        num_signals = len(self.return_file(mouse_id, treat, signal_type, date).df_list)

        if signal > num_signals:
            raise IndexError(
                "Signal {0} is out of range of the list of dataframes for the associated DataContainer".format(signal))

        return self.return_file(mouse_id, treat, signal_type, date).df_list[signal-1]


class DataContainer:

    """ Object that contains all the data from a single file """

    def __init__(self, df_list):

        self.df_list = df_list
        self.num_signals = None

        if df_list is not None: # Flag is necessary for the FeatureExtraction module
            self.num_signals = len(df_list)

    def __str__(self): # Takes care of returning the right information when DataContainer called with print()
        s = "DataContainer; num_signals: {0}".format(self.num_signals)
        return "(" + s + ")"

    def __repr__(self): # Takes care of returning the right information when DataContainer returned in shell
        s = "DataContainer; num_signals: {0}".format(self.num_signals)
        return "(" + s + ")"


class CustomDataFrame(DataMerger):
    """ Object that contains only one pd.DataFrame and offers additional functionality on this specific dataframe"""

    def __init__(self, df, id):

        self.df = df
        self.id = id

        if self.df.isna().values.any():
            warnings.warn("Data from the experiment {0} contain NAs; consider imputing them".format(id))

        if "time" not in list(df.columns):
            raise ValueError('Cannot calculate availability since there is no time column in the dataframe.')

        self.avail = [(df["time"].min(), df["time"].max())]

    def __str__(self): # Takes care of returning the right information when DataContainer called with print()
        s = 'CustomDataFrame; availability: '
        for time in self.avail:
            s += str(time) + ', '
        return "(" + s[:-2] + ")"

    def __repr__(self): # Takes care of returning the right information when DataContainer returned in shell
        s = 'CustomDataFrame; availability: ' #self.signal_type +
        for time in self.avail:
            s += str(time) + ', '
        return "(" + s[:-2] + ")"

    def right_slice(self, slice_min):
        """
        Take a right slice of the time series, i.e. cut away all the data points until minute slice_min

        :param slice_min: float
        :return: CustomDataFrame
        """

        if slice_min is None or slice_min < 0:
            raise ValueError('{0} is not a sensible value for slice_min'.format(slice_min))
        if slice_min > self.avail[0][1]:
            raise ValueError('{0} exceeds the overall length of the time series'.format(slice_min))

        data = self.df[self.df["time"] >= slice_min]

        return CustomDataFrame(df=data, id=self.id)

    def left_slice(self, slice_max):
        """
        Take a left slice of the time series, i.e. cut away all the data points from minute slice_max

        :param slice_max: float
        :return: CustomDataFrame
        """

        if slice_max is None or slice_max <= 0:
            raise ValueError('{0} is not a sensible value for slice_max'.format(slice_max))

        data = self.df[self.df["time"] <= slice_max]

        return CustomDataFrame(df=data, id=self.id)


    def partition_data(self, chunk_length, overlap_ratio, remove_shorter=True):
        """
        Chunk the data according to the chunk_length min intervals with overlap overlap_ratio. Option to include remaining
        chunks (that do not have full length) as well by setting the argument remove_shorter to False.

        :param chunk_length: float
        :param overlap_ratio: float between [0, 1)
        :param remove_shorter: boolean
        :return: list of CustomDataFrame objects
        """
        if chunk_length is None or chunk_length <= 0:
            raise ValueError('{0} is not a sensible value for chunk_length'.format(chunk_length))
        if chunk_length > self.avail[0][1]:
            raise ValueError('{0} exceeds the overall length of the time series'.format(chunk_length))
        if remove_shorter is None:
            raise ValueError('None is not a sensible value for remove_shorter')
        if 'time' not in list(self.df):
            raise ValueError('Cannot partition dataframe by time because there is no time column in it')
        if overlap_ratio is None or (overlap_ratio >= 1 or overlap_ratio < 0):
            raise ValueError('Overlap_ratio argument has to be provided and needs to be between 0 and 1 (exclusive 1)')

        diff = chunk_length * (1 - overlap_ratio)

        chunked_data = []
        start, end = self.avail[0]
        s = np.round(start)
        e = np.round(end)
        while s + chunk_length <= e:
            chunked_data.append(CustomDataFrame(self.df[(self.df['time'] >= np.round(s,1)) & (self.df['time'] < np.round(s+chunk_length, 1))], id=self.id))
            s += np.round(diff, 1)

        if not remove_shorter:
            chunked_data.append(CustomDataFrame(self.df[(self.df['time'] >= s) & (self.df['time'] < e)], id=self.id))

        return chunked_data




