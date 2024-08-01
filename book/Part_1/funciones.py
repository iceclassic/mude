import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch,find_peaks, butter, lfilter, filtfilt
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import requests
import seaborn as sns
from io import StringIO
from datetime import datetime
import plotly.graph_objects as go
#import plotly.graph_objs as go
import geopandas as gpd
import plotly

def explore_contents(data: pd.DataFrame,
                     colormap: str = 'viridis',
                     opt: dict = {'Info':True,
                                  'Time History': True,
                                  'Sparsity':True},
                    **kwargs) -> plt.figure:
    """
    Function that print a summary of th dataframe and plots the content/distribution of each column
    
     Parameters
    ----------
    data: Pandas DataFrame object, where the index is a datetime object 
    colormap: Name of the matplotlib cmap to use
    opt: Dictionary with options of different ways to explore the contents of the df
        `Info`: uses built_in methods of pandas to get column, dtype, number of entries and range of entries as basic column statistics
        `Time History`: Plots the contents of every column and the distribution of the values on it
        `Sparsity`: Heatmap of contents, plots the sparsity of each column in time
    **kwargs: Additional keyword arguments to be passed to the plotting functions ( standard matplotlib arguments such as color, alpha,etc)
      Returns:
    ----------
    Depending on the options selected in the dictionary , the function will return
        if `Info`=True -> prints a summary of the dataframe using, using method `.info`

        elif `Time History`=True-> plot with  the content/distribution of each column, 

        elif `Sparsity`=True -> plot with the sparsity of the data in time
    """

    # Make a copy of the input data
    data = data.copy()

    if opt['Info']:
        data.info()

    if opt['Time History']:
        fig, axs = plt.subplots(nrows=len(data.columns), ncols=2, figsize=(20, 3*len(data.columns)), 
                                gridspec_kw={'width_ratios': [3, 1]},**kwargs)  # Adjust the width ratio here
        plt.subplots_adjust(wspace=0.2)  

        for i, col in enumerate(data.columns):
            # Plot line 
            col_data = data[col].copy()
            col_data.dropna(inplace=True)
            if not col_data.empty:
                axs[i, 0].plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10),**kwargs)
                axs[i, 0].legend()
                axs[i, 0].set_title(str(col)+': Time Series')  # Title for the line plot
            # Plot density 
                data[col].plot.density(ax=axs[i, 1])
                axs[i, 1].set_xlim(left=data[col].min(), right=data[col].max())  # Set x-axis limits to column range
                axs[i, 1].set_ylabel('Density')
                axs[i, 1].set_title(str(col)+': Distribution')  # Title for the line plot
        plt.tight_layout()
        plt.show()

    if opt['Sparsity']:
        data.index = data.index.year
        plt.figure(figsize=(20, 10))
        sns.heatmap(data.T.isnull(), cbar=False, cmap=colormap, yticklabels=data.columns)
        plt.title('Sparsity of Time-Series')
        plt.show()


def compare_columns(data: pd.DataFrame,
                    cols: list,
                    colormap: str = 'RdYlBu',
                    norm_type: str | None = None,
                    correlation: bool = False
                    ) -> plt.figure:
    """
    Function that plot multiple columns of a DataFrame and their correlation matrix.

     Parameters
    ----------
    data: pandas.DataFrame, where the index is a datetime object
    cols: list of column names as strings to compare
    colormap: Name of the matplotlib cmap to use
    norm_type: str indicating the type of normalization, 'min_max' or 'z-norm'
    correlation: bool indicating if the correlation matrix should be plotted
    """

    # Make a copy of the input data
    data = data.copy()

    # Select only the wanted columns
    data_selected = data[cols]

    # Setup figure
    fig, axs = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(wspace=0.2)

    # Normalize the DataFrame first
    data_selected = normalize_df(data_selected, norm_type)

    for i, col in enumerate(data_selected.columns):
        # Plot line
        col_data = data_selected[col].copy()
        col_data.dropna(inplace=True)
        if not col_data.empty:
            axs.plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
            axs.legend()
            axs.set_title('Time Series')  # Title for the line plot
    plt.tight_layout()
    plt.show()


    if correlation:
        correlation_matrix = data_selected.dropna().corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm_r', fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()



def normalize_df(df: pd.DataFrame,
                 norm_type: str | None = None
                 ) -> pd.DataFrame:
    """
    Normalizes the Pandas DataFrame object.
     
     Parameters
    ----------
    df: DataFrame to normalize
    norm_type: str with the type of normalization, 'min_max' or 'z-norm'
    
    
    return: Normalized DataFrame
    """

    if norm_type is None:
        return df

    # Make a copy of the DataFrame
    df_normalized = df.copy()

    if norm_type == 'min_max':
        for column in df.columns:
            df_normalized[column] = min_max_normalization(df[column])
    elif norm_type == 'z-norm':
        for column in df.columns:
            df_normalized[column] = z_score_normalization(df[column])
    else:
        return df

    return df_normalized


def min_max_normalization(column: pd.Series) -> pd.Series:
    """
    Normalizes a pandas DataFrame Series using  min-max-normalization
    
     Parameters
    ----------
    column: Column to normalize as a pandas.Series
    
    return: The normalized column as a pandas.Series
    """
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)

    return scaled_column


def z_score_normalization(column: pd.Series) -> pd.Series:
    """
    Normalizes a pandas DataFrame Series using basic z-normalization.

     Parameters
    ----------
    column: Column to normalize as a pandas.Series

    return: The normalized column as a pandas.Series
    """

    column = pd.to_numeric(column, errors='coerce')
    mean = column.mean()
    std_dev = column.std()
    normalized_column = (column - mean) / std_dev

    return normalized_column


def filter_df(df,start_date: str | None = None,
               end_date: str | None = None,
               cols: list | None = None, 
               multiyear: list | None = None) -> pd.DataFrame:
    """ 
    Filters dataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to be filtered and converted to numpy.
    start_date : str
        The initial date for filtering the DataFrame. Format: 'MM/DD'.
   end_date : str
        The final date for filtering the DataFrame. Format: 'MM/DD'.
    multiyear : list, optional
        List of years to filter the DataFrame.
    cols : list, optional
        List of column names to filter the DataFrame. Default is None.

    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        The filtered DataFrame

 """
    df2=df.copy()
    # Ensure multiyear is a list if not provided
    if multiyear is None:
        multiyear = []

    if multiyear:
        df2 = df2[df2.index.year.isin(multiyear)]

    # Filter by month/day range if both start_date and end_date are provided
    if (start_date is not None) and (end_date is not None):
        start_date = pd.to_datetime(start_date, format='%m/%d')
        end_date = pd.to_datetime(end_date, format='%m/%d')
        mask = (df2.index.month == start_date.month) & (df2.index.day >= start_date.day) \
| (df2.index.month == end_date.month) & (df2.index.day <= end_date.day)
        

        df2 = df2[mask]

    # Select specific columns if provided
    if cols is not None:
        df2 = df2[cols]

    return df2


def plot_columns_interactive(df, column_groups: dict, title: str | None = None, 
                             xlabel: str | None = None, 
                             y_domains: dict | None = None)-> go.Figure: 
    """
    Plot columns of a DataFrame in interactive plots with multiple y-axes using Plotly.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_groups : dict
        A dictionary where keys are group names and values are lists of column names to be plotted together.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    y_domains : dict, optional
        A dictionary where keys are integers representing the y-axis index and values are lists of two floats representing the domain ( how much space each plot uses) of the y-axis.
        If None, default equidistant domains will be used based on the number of groups.
    date_focus : str, optional
        The initial focus point of the date selector buttons. Format: 'YYYY-MM-DD'.
    """
    fig = go.Figure()

    # Calculate default equidistant y-axis domains if not provided
    num_groups = len(column_groups)
    if y_domains is None:
        y_domains = {i: [i / num_groups, (i + 1) / num_groups] for i in range(num_groups)}
    
    # Add traces for each column group with separate y-axes
    for i, (group_name, columns) in enumerate(column_groups.items(), start=1):
        y_axis = f'y{i}'
        for column in columns:
            if column in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=f"{group_name}: {column}", yaxis=y_axis))
            else:
                print(f"Warning: Column '{column}' not found in DataFrame")
        
        # Update layout to add a new y-axis
        fig.update_layout(
            **{f'yaxis{i}': dict(
                title=f"{group_name}", 
                anchor='x', 
                overlaying='y', 
                side='left', 
                domain=y_domains.get(i-1, [0, 1]), 
                showline=True,
                linecolor="black",
                mirror=True,
                tickmode="auto",
                ticks="",
                titlefont={"color": "black"},
                type="linear",
                zeroline=False
            )}
        )
    
    # General layout updates
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xlabel, 
            rangeslider=dict(visible=True), 
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ),
        dragmode="zoom",
        hovermode="x",
        legend=dict(traceorder="reversed",
        x=0,
        y=1,
        xanchor='left',
        yanchor='top',
        orientation='v'
    ),
        height=800,
        template="plotly",
        margin=dict(t=90, b=150)
    )

    # Add break up times shapes if necessary
    break_up_times = pd.read_csv('https://raw.githubusercontent.com/iceclassic/sandbox/main/Data/BreakUpTimes.csv')
    break_up_times['timestamp'] = pd.to_datetime(break_up_times[['Year', 'Month', 'Day']])
    break_up_times.set_index('timestamp', inplace=True)
    shapes = []
    for date in break_up_times.index:
        shape = {"type": "line", "xref": "x", "yref": "paper", "x0": date, "y0": 0, "x1": date, "y1": 1,
                 "line": {"color": 'red', "width": 0.6, "dash": 'dot'}, 'name': 'break up times'}
        shapes.append(shape)

    fig.update_layout(shapes=shapes)

    fig.show()


def seasonal_trends(df, columns_to_plot=None, k=1, plot_mean_std=False, historicalVariation=False,
                     multiyear=None, Compare_years_to_baseline=False, holdPlot=False, xaxis='Days since start of year',xlim=None,**kwargs):
    """
    Plot the yearly distribution of temperature data for specified columns.

    Parameters:
    df (DataFrame): The input DataFrame containing temperature data with a datetime index.
    columns_to_plot (list, optional): List of column names to plot. 
                                      If None, plot all columns except xaxis column.
    k (int, optional): Number of standard deviations to plot around the average.
    plot_mean_std (bool, optional): Whether to plot the mean and standard deviation. Default is True.
    historicalVariation (bool, optional): Whether to use different colors for each year's data. Default is False.
    multiyear (list or None, optional): The list of years to consider for filtering the data. 
                                        If None, all years are considered. Default is None.
    Compare_years_to_baseline (bool, optional): Compare specific years to a baseline year. Default is False.
    holdPlot (bool, optional): Whether to hold the plot and not display it. Default is False.
    xaxis (str, optional): Column name for x-axis. Default is "Days since start of year".
    xlim (list, optional): limit to the xaxis when plotting
    **kwargs: Additional keyword arguments to be passed to the plotting functions ( standard matplotlib arguments such as color, alpha,etc)
    Returns:
    None
    """
    if columns_to_plot is None:
        columns_to_plot = [col for col in df.columns if col != xaxis]

    num_plots = len(columns_to_plot)
    fig, ax = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))

    if num_plots == 1:
        ax = [ax]  # Make ax iterable

    if Compare_years_to_baseline:
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(min(multiyear), max(multiyear))
    elif historicalVariation:
        years = df.index.year.unique()
        cmap = plt.get_cmap('viridis', len(years))
        norm = plt.Normalize(min(years), max(years))

    for i, col in enumerate(columns_to_plot):
        df_nonan = df[[col, xaxis]].dropna()
        df_nonan['Year'] = df_nonan.index.year

        average = df_nonan.groupby(xaxis)[col].mean()
        std = df_nonan.groupby(xaxis)[col].std()

        if Compare_years_to_baseline:
            for year in multiyear:
                if year in df_nonan['Year'].unique():
                    year_data = df_nonan[df_nonan['Year'] == year]
                    year_data = year_data.sort_values(by=xaxis)  # useful when xaxis spans multiple years to avoid lines crossing whe using plot
                    ax[i].plot(year_data[xaxis], year_data[col], color=cmap(norm(year)))
                else:
                    print(f"No {col} data available for year {year}")
        elif historicalVariation:
            for year in years:
                year_data = df_nonan[df_nonan['Year'] == year]
                ax[i].scatter(year_data[xaxis], year_data[col], marker='.', color=cmap(norm(year)))

        ax[i].scatter(df_nonan[xaxis], df_nonan[col], marker='.', label=col,**kwargs)

        if plot_mean_std:
            ax[i].plot(average.index, average, color='b', label=f'mean ±{k} std')
            ax[i].fill_between(average.index, average + k * std, average - k * std, color='b', alpha=0.2)

        ax[i].set_ylabel(f'{col}')
        ax[i].set_title(f'{col}')
        ax[i].legend()
        ax[i].set_xlabel(f'Days since {xaxis}')
        ax[i].set_xlim(xlim)

        if Compare_years_to_baseline or historicalVariation:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax[i])
            cbar.set_label('Year')

    plt.tight_layout()
    if not holdPlot:
        plt.show()


def compute_and_plot_psd(df, cols=None, nperseg=None, plot_period=False, apply_filter=False, max_allowed_freq=None,
                         filter_order=4, find_peaks_kwargs=None):
    """
    Compute and plot the Power Spectral Density (PSD) for the specified columns in the DataFrame.
    If no columns are specified, compute and plot the PSD for all columns.
    Optionally apply a low-pass filter to the data before computing the PSD.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    cols (list or None): List of column names to compute the PSD for. If None, all columns are used.
    nperseg (int or None): Length of each segment for Welch's method. Default is None, which uses the default of `scipy.signal.welch`.
    plot_period (bool): Whether to plot the period (True) or frequency (False) on the x-axis.
    apply_filter (bool): Whether to apply a low-pass filter to the data. Default is False.
    max_allowed_freq (float or None): Maximum allowed frequency for the low-pass filter. Required if apply_filter is True.
    filter_order (int): The order of the Butterworth filter. Default is 4.
    find_peaks_kwargs (dict or None): Additional keyword arguments to be passed to `find_peaks`.

    Returns:
    dict: Dictionary containing the PSD values, frequencies, peak information, and period for each column.
    """
    if find_peaks_kwargs is None:
        find_peaks_kwargs = {}

    if cols is None:
        cols = df.columns

    nyquist_freq = 0.5  # Nyquist frequency for a sampling rate of 1 day
    plt.figure(figsize=(20, 10))
    
    psd_dict = {}

    for col in cols:
        if col in df.columns:
            # Drop NaN values to handle different ranges of data
            valid_data = df[col].dropna()
            
            if len(valid_data) == 0:
                print(f"No valid data for column '{col}'. Skipping.")
                continue
            
            # Apply low-pass filter if requested
            if apply_filter:
                if max_allowed_freq is None:
                    raise ValueError("max_allowed_freq must be specified if apply_filter is True.")
                if max_allowed_freq > nyquist_freq:
                    raise ValueError(f"max_allowed_freq must be <= {nyquist_freq}")
                
                # Design a Butterworth filter
                b, a = butter(filter_order, max_allowed_freq, btype='low', analog=False, fs=1.0)
                valid_data = filtfilt(b, a, valid_data)
            
            # Compute the PSD using a sampling frequency of 1 day (fs = 1)
            f, Pxx = welch(valid_data, fs=1.0, nperseg=nperseg if nperseg else len(valid_data)//2)
            
            # Filter out frequencies higher than the Nyquist frequency
            valid_indices = f <= nyquist_freq
            f = f[valid_indices]
            Pxx = Pxx[valid_indices]
            
            if plot_period:
                # Convert frequency to period
                with np.errstate(divide='ignore'):
                    x_values = np.where(f == 0, np.inf, 1 / f)  # Convert frequencies to periods, avoiding division by zero
                
                # Filter out infinite and NaN periods
                valid = np.isfinite(x_values) & ~np.isnan(Pxx)
                x_values = x_values[valid]
                Pxx = Pxx[valid]
                x_label = 'Period [days]'
            else:
                x_values = f
                x_label = 'Frequency [cycles/day]'
            
            # Find peaks in the PSD
            peaks, _ = find_peaks(Pxx, **find_peaks_kwargs)
            peak_freqs = f[peaks]
            peak_psd_values = Pxx[peaks]
            peak_periods = 1 / peak_freqs  # Calculate periods in days
            
            # Store PSD values and peak information in the dictionary
            psd_dict[col] = {
                'frequencies': f,
                'psd_values': Pxx,
                'peak_frequencies': peak_freqs,
                'peak_psd_values': peak_psd_values,
                'peak_periods': peak_periods
            }
            
            # Plotting
            plt.plot(x_values, Pxx, label=col)
    
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('PSD [unit^2/day]')
    plt.title('PSD of Selected Columns')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.ylim(10e-10,10e10)
    if plot_period:
        plt.xlim(1, 800)
    else:
        plt.xlim(0, max_allowed_freq)
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.show()

    return psd_dict

def compute_and_plot_fourier(df, cols=None, plot_period=False):
    """
    Compute and plot the Fourier Transform for the specified columns in the DataFrame.
    If no columns are specified, compute and plot the Fourier Transform for all columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    cols (list or None): List of column names to compute the Fourier Transform for. If None, all columns are used.
    plot_period (bool): Whether to plot the period (True) or frequency (False) on the x-axis.

    Returns:
    dict: Dictionary containing the Fourier Transform values and frequencies for each column.
    """
    if cols is None:
        cols = df.columns

    plt.figure(figsize=(20, 10))
    fft_dict = {}

    for col in cols:
        if col in df.columns:
            # Drop NaN values to handle different ranges of data
            valid_data = df[col].dropna()

            if len(valid_data) == 0:
                print(f"No valid data for column '{col}'. Skipping.")
                continue

            # Compute the Fourier Transform
            ft = fft(valid_data)
            freq = fftfreq(len(valid_data))

            # Store Fourier Transform values and frequencies in the dictionary
            fft_dict[col] = {
                'values': ft,
                'frequencies': freq
            }

            # Plotting
            if plot_period:
                # Convert frequency to period
                with np.errstate(divide='ignore'):
                    x_values = np.where(freq == 0, np.inf, 1 / freq)  # Convert frequencies to periods, avoiding division by zero
                x_label = 'Period [days]'
            else:
                x_values = freq
                x_label = 'Frequency [cycles/day]'

            plt.plot(x_values, np.abs(ft), label=col)

    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Fourier Transform')
    plt.title('Fourier Transform of Selected Columns')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    if plot_period:
        plt.xlim(1, 400)
        plt.legend(loc='upper left')
    plt.minorticks_on()
    plt.show()

    return fft_dict

def import_data_browser(url):
    """
    This function imports data from a specified URL.

    Parameters:
    url (str): The URL from which to import the data.

    Returns:
    None

    Comments:
    This function is needed to load data in a browser, as the environment used does not support absolute/relative path imports 
    """
   
    response = requests.get(url)
    csv_data = StringIO(response.text)

    return csv_data

def days_since_last_date(df, date, name=None):
    """
    Calculate the number of days since the last occurrence of a given month and day or a list of dates.

    Parameters:
    - df: DataFrame
        The DataFrame containing the dates.
    - date_or_dates: str or list of str
        A single date in the format 'MM/DD' or a special date keyword, or a list of dates in the format 'YYYY/MM/DD'.
    - name: str, optional
        The name of the column to add when using a single date. If None, defaults to the month_day or special date keyword.

    Returns:
    - df: DataFrame
        The DataFrame with additional columns containing the number of days since the last occurrence of the given date(s).
    """
    df = df.copy()

    # Function to calculate days since last occurrence of given month and day
    def days_since(date, target_date):
        """
        Calculate the number of days since a given date.

        Parameters:
        date (datetime): The date to calculate the number of days since.
        target_date (datetime): The target date to calculate the number of days since.

        Returns:
        int: The number of days since the given date.
        """
        this_year = date.year
        target_date_this_year = datetime(this_year, target_date.month, target_date.day)
        target_date_last_year = datetime(this_year - 1, target_date.month, target_date.day)

        # Calculate difference
        if date >= target_date_this_year:
            days_diff = (date - target_date_this_year).days
        else:
            days_diff = (date - target_date_last_year).days

        # If days_diff is negative, it means the target date has not occurred this year yet, so we use the previous year's target date
        if days_diff < 0:
            days_diff += 365

        return days_diff

    # date or a special date keyword
    if isinstance(date, str):
        if date == 'Summer Solstice':
            month, day = 6, 21
        elif date == 'Winter Solstice':
            month, day = 12, 21
        elif date == 'Spring Equinox':
            month, day = 3, 21
        elif date == 'Fall Equinox':
            month, day = 9, 21
        else: # string wiht date  'MM/DD' ( cannot have years as this loop computes the date for each year)
            month, day = map(int, date.split('/'))
        
        target_date = datetime(df.index[0].year, month, day)
        column_name = name if name else date
        df[column_name] = df.index.map(lambda x: days_since(x, target_date))

    # list of dates
    elif isinstance(date, list):
        # Convert date strings to datetime objects
        past_dates = [datetime.strptime(date_str, '%Y/%m/%d') for date_str in date]

        # Function to find the closest past date and calculate days since
        def closest_past_days(current_date):
            # Filter for valid past dates
            valid_dates = [d for d in past_dates if d <= current_date]
            
            # If no valid dates, return None or desired value
            if not valid_dates:
                return None
            
            # Find the closest past date
            closest_date = max(valid_dates)
            return (current_date - closest_date).days

        column_name = name if name else 'days_since_closest_date'
        df[column_name] = df.index.map(closest_past_days)


    return df


def plot_interactive_map():
    
    plotly.offline.init_notebook_mode()




    # Define click event handler
    def click_callback(trace, points, selector):
        if points:
            lat = points.xs[0]
            lon = points.ys[0]
            print(f"Latitude: {lat}, Longitude: {lon}")

    # Latitude and longitude coordinates for Nenana, Alaska
    nenana_lat = 64.56702898502982
    nenana_lon = -149.0815700675435

    USW00026435_NENANA_LAT=64.54725
    USW00026435_NENANA_LOG=	-149.08713

    USW00026435_Fairbanks_LAT=64.80309
    USW00026435_Fairbanks_LOG=	-147.87606

    square_lat = [64, 64, 65, 65,64]  # Latitude of vertices
    square_lon = [-150, -149, -149, -150,-150]  # Longitude of vertices

    gulkana_lat= 63.2818
    gulkana_lon=-145.426 

    usgs_tenana_river_lat=64.5649444
    usgs_tenana_river_lon=-149.094 

    usgs_tenana_fairbanks_lat=64.792344 
    usgs_tenana_fairbanks_lon=-147.8413097 

    square_lat_w = [64, 64, 66, 66,64]  # Latitude of vertices
    square_lon_w = [-151, -149, -149, -151,-151]  # Longitude of vertices

    #hydrographic_gdf = gpd.read_file('../../data/shape_files/hybas_na_lev01_v1c.shp')

    # Initialize Plotly figure with initial zoom level and center coordinates
    fig = go.Figure()

    # Add markers
    fig.add_trace(go.Scattermapbox(
        lat=[nenana_lat],
        lon=[nenana_lon],
        mode='markers',
        marker=dict(
            size=10,
            color='purple',
            opacity=0.8  # Increase opacity
        ),
        text=["NENANA tripod"],  # Text label for the marker
        hoverinfo="text" , # Only show text on hover
        name="Ice classic tripod"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[USW00026435_NENANA_LAT],
        lon=[USW00026435_NENANA_LOG],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            opacity=0.8  # Increase opacity
        ),
        text=["USGS Weather Station USW00026435"],  # Text label for the marker
        hoverinfo="text",  # Only show text on hover
        name="Nenana Weather Station"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[USW00026435_Fairbanks_LAT],
        lon=[USW00026435_Fairbanks_LOG],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            opacity=0.8  # Increase opacity
        ),
        text=["USGS Weather Station USW00026411"],  # Text label for the marker
        hoverinfo="text",  # Only show text on hover
        name="Fairbanks Weather Station"
    ))
    fig.add_trace(go.Scattermapbox(
        lat=square_lat,
        lon=square_lon,
        mode='lines',  # Draw lines between vertices
        line=dict(color='yellow'),  # Color of the lines
        fill='toself',  # Fill the inside of the polygon
        fillcolor='rgba(255, 239,0, 0.1)',
        name='Temperature',
        text="Berkeley Earth Global",
        hoverinfo='text'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[gulkana_lat],
        lon=[gulkana_lon],
        mode='markers',  # Draw lines between vertices
        marker=dict(
            size=10,
            color='blue',
            opacity=0.8  # Increase opacity
        ),
        name='Gulkana  Glacier',
        text="USGS  Weather Station  15485500",
        hoverinfo='text'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[usgs_tenana_river_lat],
        lon=[usgs_tenana_river_lon],
        mode='markers',  # Draw lines between vertices
        marker=dict(
            size=10,
            color='green',
            opacity=0.8  # Increase opacity
        ),
        name='Tenana R at Nenana',
        text="USGS  Weather Station 15515500",
        hoverinfo='text'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[usgs_tenana_fairbanks_lat],
        lon=[usgs_tenana_fairbanks_lon],
        mode='markers',  # Draw lines between vertices
        marker=dict(
            size=10,
            color='green',
            opacity=0.8  # Increase opacity
        ),
        name='Tenana R at Fairbanks',
        text="USGS  Weather Station 15515500",
        hoverinfo='text'
    ))
    fig.add_trace(go.Scattermapbox(
        lat=square_lat_w,
        lon=square_lon_w,
        mode='lines',  # Draw lines between vertices
        line=dict(color='pink'),  # Color of the lines
        fill='toself',  # Fill the inside of the polygon
        fillcolor='rgba(255, 20,147, 0.01)',
        name='Solar Radiation and Cloud Coverage',
        text="TEMIS & NERC-EDS",
        hoverinfo='text'
    ))
    # fig.add_trace(go.Scattermapbox(
    #     lat=hydrographic_gdf.geometry.y,
    #     lon=hydrographic_gdf.geometry.x,
    #     mode='markers',
    #     marker=dict(
    #         size=8,
    #         color='blue',
    #         opacity=0.6
    #     ),
    #     name='Hydrographic Information',
    #     text=hydrographic_gdf['name'],  # Assuming there's a 'name' column
    #     hoverinfo='text'
    # ))



    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=nenana_lat, lon=nenana_lon),
            zoom=5
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # Set margins to zero
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            orientation='v'
        ),
        updatemenus=[  # Add buttons for layer toggling
            dict(
                buttons=list([
                    dict(
                        args=['visible', [False, False, False, True, True, True, True, True, False, False]],
                        label='Hide (some) Layers',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [True, True, True, True, True, True, True, True, True, True]],
                        label='Show (all) Layers',
                        method='restyle'
                    )
                ]),
                direction='left',
                pad={'r': 10, 't': 10},
                showactive=True,
                type='buttons',
                x=0.1,
                xanchor='left',
                y=1.1,
                yanchor='top'
            ),
        ]
    )

    # Add invisible scatter plot trace to capture click events
    click_trace = go.Scattermapbox(lat=[], lon=[], mode='markers', marker=dict(opacity=0))
    fig.add_trace(click_trace)

    # Update click event handler
    click_trace.on_click(click_callback)


    # Show interactive plot
    fig.show()