import datetime as dt
import pytz


tz = pytz.timezone('Europe/Brussels')


def dt_from_date(date, end=False, timestamp=False):
    """
    Returns the datetime corresponding to the first moment of `date`: 0:00. If
    `end` is True, returns the first moment of the next day. Furthermore, the
    returned datetime is localized according to the Belgian timezone. If
    timestamp is true, returns the timestamp instead.

    Arguments
    =========
     - date: datetime.date
        day considered
     - end: bool
        whether to consider the following day beginning
     - timestamp: bool
        whether to convert the datetime to a timestamp
    """
    d = dt.datetime.combine(date, dt.time(0, 0)) + end * dt.timedelta(days=1)
    d = tz.localize(d)
    if timestamp:
        return int(d.timestamp())
    return d


def dt_from_formatted_date(formatted, timestamp=False):
    """
    Convert a date under the format "dd/mm/yyyy hh:mm" to a datetime localized
    according to the Belgian timezone. If timestamp is true, returns the
    timestamp instead.

    Arguments
    =========
     - formatted: string
        date to parse
     - timestamp: bool
        whether to convert the datetime to a timestamp
    """
    d = tz.localize(dt.datetime.strptime(formatted, '%d/%m/%Y %H:%M'))
    if timestamp:
        return int(d.timestamp())
    return d


def set_time_index(dataframe, index_col, verbose=False):
    """
    Transforms `dataframe` by setting its new index on a timestamp columns
    `index_col`. This function detects and correct the duplicated timestamp
    produced by a naive parsing of the dates at DST change from summer to
    winter time.

    NB: this function assume a most ONE DST change in `dataframe`

    Arguments
    =========
     - dataframe: pandas.DataFrame
        the dataframe to index
     - index_col: string
        the label of the column to use as index
    """
    duplicated_dt = dataframe[index_col].duplicated(keep='first')
    if duplicated_dt.sum() == 4:
        dataframe.loc[duplicated_dt, index_col] -= 60 * 60
        if verbose:
            print('DST change detected and corrected')

    dataframe.set_index(index_col, inplace=True)

    return dataframe


def date_from_ts(timestamp):
    """
    Returns the date of `timestamp`.

    Arguments
    =========
     - timestamp: float or int
        timestamp of the date requested
    """
    return dt.datetime.fromtimestamp(timestamp).date()
