import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import pandas as pd
from scipy.signal import welch,find_peaks, butter, lfilter, filtfilt
from scipy.fft import fft, fftfreq
from datetime import time ,datetime,timedelta
from scipy import stats
import requests
import seaborn as sns
from io import StringIO
import geopandas as gpd



def finite_differences(series:pd.Series):
    """
    Computes forward, central, and backward differences using the step size as days between measurement.
    
    Parameters
    ---------
    series: pd.Series
        Series with datetime index 
    
    Returns
    ---------
    df: pd.DataFrame
        DataFrame with forward, backward and central differences for the Series
    
    """
   
    days_forward = (series.index.to_series().shift(-1) - series.index.to_series()).dt.days
    days_backward = (series.index.to_series() - series.index.to_series().shift(1)).dt.days
    
    # Forward difference: (f(x+h) - f(x)) / h 
    forward = (series.shift(-1) - series) / days_forward

    # Backward difference: (f(x) - f(x-h)) / h,
    backward = (series - series.shift(1)) / days_backward

    # Central difference: (f(x+h) - f(x-h)) / (h_forward + h_backward)
    central = (series.shift(-1) - series.shift(1)) / (days_forward + days_backward) #
    
    # fixing start/end points
    forward.iloc[-1] = np.nan  
    backward.iloc[0] = np.nan  

    return pd.DataFrame({'forward': forward, 'backward': backward, 'central': central})

def plot_gradients(ax:plt.axes,
                    x:pd.Series, 
                    y:pd.Series,
                    slopes:pd.Series,
                    length: int= 2,
                    color: str='blue',
                    label:str=None,
                    label_flag: bool=True,
                    offset: int=0 ):
    """
    Plot gradient slopes and annotate them.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x : pd.Series
        The x-values of the data points.
    y : pd.Series
        The y-values of the data points.
    slopes : pd.Series
        The slopes of the data points. Each column column corresponds to a different gradient type.
    length : int, optional
        The length of the gradient line. Default is 2.
    color : str, optional   
        The color of the gradient line. Default is 'blue'.
    label : str, optional
        The label of the gradient line. Default is None.
    label_flag : bool, optional
        Whether to show the label. Default is True.
    offset : int, optional
        The offset of the annotation. Default is 0. Use to avoid overlapping annotations.

    Returns
    -------
    annotations : list
        A list of annotations of the value of the gradient at each point, using all the graudient types.
    """
    #
    # (annotations)code shamelessly stolen from chatgpt 
    annotations = []
    for i in range(len(x)):
        if np.isnan(slopes.iloc[i]):
            continue  
        dx = length  
        dy = slopes.iloc[i] * dx  
        
        ax.plot(
            [x[i], x[i] + pd.Timedelta(days=dx)], 
            [y.iloc[i], y.iloc[i] + dy], 
            color=color, alpha=0.7, label=label if label_flag else ""
        )
        
        annotations.append({
            'x': x[i] + pd.Timedelta(days=dx / 2),
            'y': y.iloc[i] + offset,
            'text': f'{slopes.iloc[i]:.2f}',
            'color': color
        })
        label_flag = False  #


    for annotation in annotations:
        ax.annotate(
            annotation['text'], 
            (annotation['x'], annotation['y']), 
            color=annotation['color'], fontsize=10, ha='center'
        )

def plot_gradients_and_timeseries(result:pd.DataFrame,
                                 col:pd.Series, 
                                 year:int, 
                                 plot_gradient_as_slope:bool=False,
                                 Title: str=None ,
                                 ylabel: str=None,
                                 xlim: list=['01/01', '12/31'],
                                 ylim: list=None,
                                 annotation_offsets: dict= {'forward': -3, 'backward': -6, 'central': -9},
                                 vline: dict=None):
    """
    Plots ice thickness gradients and ice thickness for a selected year.

    Parameters
    ----------
    result: pd.DataFrame
        DataFrame containing the gradients with columns ['forward', 'backward', 'central'].
    col: pd.Series
         Series containing the columns values indexed by datetime.
    year: int
        The year to filter and plot data for.
    plot_gradient_as_slope: bool
        If True, plots lines with gradients and annotations.
    Title: str
        title of the plot
    ylabel: str
        ylabel of the plot
    ylim: list
        y-axis limits for the plot. List should be in the format [(MM/DD),(MM/DD)].
    annotation_offsets: dict
        The offset of the annotation. Default value assume that the series has three column corresponding to 
        forward, backward and central. Use to avoid overlapping annotations. The value corresponds to the y-axis value.
    vline: dict
        The vertical lines to plot. The key is the label and the value(str) is the date in the format MM/DD.
        
    Returns
    -------
    plt.figure
    """
    
    result_year = result[result.index.year == year]
    col_year = col[col.index.year == year]

    xlimits=pd.to_datetime([str(year) + '/' + e for e in xlim])

    
    if not plot_gradient_as_slope:
        fig, ax1 = plt.subplots(figsize=(20, 5))

        ax1.plot(result_year['forward'], label='Forward', color='red')
        ax1.plot(result_year['backward'], label='Backward', color='green')
        ax1.plot(result_year['central'], label='Central', color='blue')

        ax1.set_ylabel(ylabel)
        ax1.set_xlabel('Date')
        ax1.set_title(Title)
        ax1.grid()


        ax1.set_xlim(xlimits)
        ax1.set_ylim(ylim)

        if vline is not None:
            for key, value in vline.items():
                ax1.axvline(pd.to_datetime(str(year) + '/' + value), color='magenta', linestyle='--', label=key,linewidth=3)	

      
        #ax1.set_xlim(result_year.index.min(), result_year.index.max())
        ax1.legend(loc='upper left')

      
        ax2 = ax1.twinx()
        ax2.scatter(col_year.index, col_year, color='black', alpha=0.4, label='data')
        ax2.plot(col_year.index, col_year, color='black', alpha=0.4)
        

       
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(20, 5))

        ax.plot(col_year.index, col_year, label='Thickness', color='black', linestyle='--', alpha=0.4)
        ax.scatter(col_year.index, col_year, color='black', alpha=0.4)

      
        offsets = annotation_offsets
        
        for grad_type, color in zip(['forward', 'backward', 'central'], ['red', 'green', 'blue']):
            slopes = result_year[grad_type]
            plot_gradients(ax, result_year.index, col_year, slopes, color=color, label=grad_type.capitalize(), label_flag=True, offset=offsets[grad_type])

        ax.set_title(Title)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlimits)
        ax.set_ylim(ylim)
        ax.grid(True)
        if vline is not None:
            for key, value in vline.items():
                ax.axvline(pd.to_datetime(str(year) + '/' + value), color='magenta', linestyle='--', label=key,linewidth=3)
        
        ax.legend()
        plt.show()





def explore_contents(data: pd.DataFrame,
                     colormap: str = 'viridis',
                     opt: dict = {'Info':True,
                                  'Time History': True,
                                  'Sparsity':True},
                    **kwargs) -> plt.figure:
    """

    Function that prints a summary of the dataframe and plots the content/distribution of each column
    
    Parameters
    ----------
    data: pd.DataFrame
        DataFrames with datetime index 
    colormap: str
        Name of the matplotlib cmap to use in correlation matrix 
    opt: dict
        Dictionary with options of different ways to explore the contents of the df
            `Info`: uses built_in methods of pandas to get column, dtype, number of entries and range of entries as basic column statistics
            `Time History`: Plots the contents of every column and the distribution of the values on it
            `Sparsity`: Heatmap of contents, plots the sparsity of each column in time
    **kwargs: Additional keyword arguments to be passed to each of the timeseries plot ( standard matplotlib arguments such as color, alpha,etc)
    
    Returns
    ----------
    Depending on the options selected in the dictionary , the function will return:

        -`Info`=True -> prints a summary of the dataframe using, using method `.info`
    
        -`Time History`=True-> plot with  the content/distribution of each column, 

        -`Sparsity`=True -> plot with the sparsity of the data in time
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
                data[col].plot.density(ax=axs[i, 1],color=plt.cm.tab10(i % 10))
                axs[i, 1].set_xlim(left=data[col].min(), right=data[col].max())  # Set x-axis limits to column range
                axs[i, 1].set_ylabel('Density')
                axs[i, 1].set_title(str(col)+': Distribution')  # Title for the line plot
        fig.tight_layout()
        

    if opt['Sparsity']:
        data.index = data.index.year
        plt.figure(figsize=(20, 10))
        sns.heatmap(data.T.isnull(), cbar=False, cmap=colormap, yticklabels=data.columns)
        plt.title('Sparsity of Time-Series')
        plt.show()


def compare_columns(df: pd.DataFrame,
                    columns:list,
                    colormap: str = 'RdYlBu',
                    norm_type: str | None = None,
                    correlation: bool = False,
                    **kwargs
                    ) -> plt.figure:
    """
    Simple function thats plot multiple columns of a DataFrame in a single plot.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with datetime index
    columns: list of str 
        Names of the columns to visually  compare
     colormap: str
        Name of the matplotlib cmap to use in correlation matrix plot
    norm_type: str 
        Indicates if the values are normalized, allowable values are `None`, `min_max` or `z-norm`
    correlation: bool 
        Indicating if the correlation matrix should be plotted
    """
    
    fig, axs = plt.subplots(1, ncols=2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 2]})
    plt.subplots_adjust(wspace=0.1)

    # Normalize the DataFrame
    data_copy=df.copy()
    data=data_copy[columns]
    data= normalize_df(data, norm_type)

    # Plot the time series on the first subplot (axs[0])
    axs[0].plot(data.index,data.values)
    axs[0].set_title('Time Series')
    axs[0].legend(columns)


    data.plot.density(ax=axs[1])
    axs[1].set_ylabel('Density')
    axs[1].set_title(f'Distribution')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return fig


    if correlation:
        correlation_matrix = data.dropna().corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm_r', fmt=".2f", linewidths=0.5, vmin=0, vmax=1)
        plt.title('Correlation Matrix')
        plt.show()
        return fig


def normalize_df(df: pd.DataFrame | pd.Series,
                 norm_type: str | None = None,
                 columns:list|None=None
                 ) -> pd.DataFrame:
    """
    Normalizes the Pandas DataFrame object.
     
     Parameters
    ----------
    df: DataFrame to normalize
    norm_type: str with the type of normalization, `min_max` or `z-score`.
    
    
    return: Normalized DataFrame
    """

    if norm_type is None:
        return df

    # Make a copy of the DataFrame
    df_normalized = df.copy()
    if isinstance(df, pd.Series):
        if norm_type == 'min_max':
            df_normalized= min_max_normalization(df)
        elif norm_type == 'z-norm':
                df_normalize = z_score_normalization(df)
    else:
      for col in columns:
            if norm_type == 'min_max':
                df_normalized[col] = min_max_normalization(df[col])
            elif norm_type=='z-score':
                df_normalized[col] = z_score_normalization(df[col])
            else:
                raise ValueError('normalization method not implemented')
   
    return df_normalized


def min_max_normalization(column: pd.Series) -> pd.Series:
    """
    Normalizes a pandas DataFrame Series using  min-max-normalization
    
     Parameters
    ----------
    column: Column to normalize
    
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
        The input DataFrame to be filtered
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
                             xlabel: str | None = 'Date', 
                             y_domains: dict | None = None)-> go.Figure: 
    """
    Plot columns of a DataFrame in interactive plots with multiple y-axes using Plotly.

    Parameters
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    column_groups : dict
        A dictionary where keys are group names and values are lists of column names to be plotted together.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    date_focus : str, optional
        The initial focus point of the date selector buttons. Format: 'YYYY-MM-DD'.

    Returns
    -------
    fig : plotly.graph_objs.Figure
    """
    fig = go.Figure()
    
    num_groups = len(column_groups)
    y_domains = {i: [i / num_groups, (i + 1) / num_groups] for i in range(num_groups)}

    # Add traces for each column group with separate y-axes
    for i, (group_name, columns) in enumerate(column_groups.items(), start=1):
        y_axis = f'y{i}'
        for column in columns:
            if column in df.columns:
                col_data = df[column].copy()
                col_data.dropna(inplace=True)
                fig.add_trace(go.Scatter(x=col_data.index, y=col_data, mode='lines', name=f"{group_name}: {column}", yaxis=y_axis))
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
    break_up_times = pd.read_csv('../../Data/BreakUpTimes.csv')
    break_up_times['timestamp'] = pd.to_datetime(break_up_times[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    break_up_times.set_index('timestamp', inplace=True)
    shapes = []
    for date in break_up_times.index:
        shape = {"type": "line", "xref": "x", "yref": "paper", "x0": date, "y0": 0, "x1": date, "y1": 1,
                 "line": {"color": 'red', "width": 0.6, "dash": 'dot'}, 'name': 'break up times'}
        shapes.append(shape)

    fig.update_layout(shapes=shapes)
    # dumm line to add to legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],  
        mode='lines',
        line=dict(color='red', width=0.6, dash='dot'),
        name='Break Up Times',  
        hoverinfo='none',  \
        showlegend=True     
    ))
    #fig.show()
    return fig


def plot_contents(  
    df: pd.DataFrame,
    columns_to_plot: list[str] | None = None,
    k: int = 1,
    plot_mean_std: bool = False,
    multiyear: list[int] | None = None,
    plot_together: bool = False,
    xaxis: str = 'Days since start of year',
    xaxis_name: str | None = None,
    xlim: list[float] | None = None,
    col_cmap: str = 'Set1',
    years_cmap: str = 'viridis',
    scatter_alpha: float = 0.1,
    std_alpha: float = 0.3,
    ylim: list[float] | None = None,
    years_line_width: int = 4,
    plot_break_up_dates:bool=False,
    normalize:str| None=None,
    Title:str|None=None,
    y_label:str|None=None,
):
    """
    Plots the data for the specified columns.

    Parameters
    ----------
    df: pd.DataFrame:
        The input DataFrame with a datetime index.

    columns_to_plot: list, optional
        List of column names to plot. If None, plot all columns except xaxis column.

    plot_together: bool, optional
        if `True`, plot all specified columns together on a single plot. Default is `False`.

    multiyear: list, optional
        The list of years to consider for filtering the data. Default is None. If None, all years are considered.

    plot_mean_std: str, optional
        Whether to plot the mean and standard deviation. Default is  True.
            `True`, mean and standard deviation plot are plotted on top of scatter plot 
            `'only'`, the scatter points are not plotted, 

    k: int, optional
        Number of standard deviations to plot around the average. Default is `k=1` 

    xaxis: str, optional
        Column name for x-axis. Default is `xaxis="Days since start of year"`.
        If `xaxis='index'`, the index of the df is used ( the index should be a datetime object). This essentially recovers the timeseries plot. 
    
    xlim: list, optional
        Limits to the x-axis when plotting. 
        if `xaxis='index'`, the expected format is `['YYYY/MM/DD', 'YYYY/MM/DD']`.

    col_cmap: str, optional
        	Sequential colormap to use for plotting different columns (name of matplotlib cmaps). Default is 'Set1'.
            if a list of colors is passed with the same length as the number of column, it used that list of color.

    year_map: str, optional
        Sequential colormap to use for plotting different years (name of matplotlib cmaps). Default is 'viridis'.

    scatter_alpha: float, optional
        Opacity (Alpha value) for the scatter plot. Default is 0.01.

    std_alpha: float, optional
        Opacity (Alpha value) for the  fill area delimited by the standard deviation plot. Default is 0.3.

    ylim: list, optional
        Limit of y-axis when plotting. Each element is list with the limits for each column.
    
    years_line_width: int, optional
        Line width for the plot of a specific year. Default is 5.

    plot_break_up_dates:bool
        Wether we plot scatter point associated with break up.Only if xaxis='day_of_year'. Not yet available with `plot_together=True` , 
        as it create multiple equal scatter points. It also annotated each scatter point with year

    normalize: str,optional
        if  `plot_together=True`, normalization can be applied in order to plot them together.
        The normalization can be `min_max` or `z-score`. Default is None.
    Title: str,optional
        if plot_together=True, the title of the plot
    y_label:str,optional
        if plot_together=True, the label of the y-axis
    Returns:
    ----------
    fig(s) : plotly.graph_objs.Figure
    """

    # basic functionally
    if columns_to_plot is None:
        columns_to_plot = [col for col in df.columns if col != xaxis]
    if multiyear is None:
        compare_years_to_baseline = False
    else:
        compare_years_to_baseline =True

    if xaxis_name is None:
        xaxis_name = xaxis
    if plot_together:
        fig, ax = plt.subplots(figsize=(20, 5))

    else:
        num_plots = len(columns_to_plot)
        fig, ax = plt.subplots(num_plots, 1, figsize=(20, 5 * num_plots))
        if num_plots == 1:
            ax = [ax]  # Make ax iterable
    
    
    df_break_up=df[df['Days until break up']==0] # we extract break up dates
    if isinstance(multiyear,list):
        df_break_up=df_break_up[df_break_up.index.year.isin(multiyear)]

    # colors
    if isinstance(col_cmap,str):
        seq_map = plt.get_cmap(col_cmap)
    if isinstance(col_cmap,list):
        seq_map=mcolors.ListedColormap(col_cmap)

    colors = seq_map(np.linspace(0,1, len(columns_to_plot))) 


    if compare_years_to_baseline:
        if len(multiyear) == 1:
           # single_year = multiyear[0]
            single_color = plt.get_cmap(years_cmap)(0.5)  
            cmap = plt.cm.colors.ListedColormap([single_color])  
            norm = plt.Normalize(0, 1) 
        else:
            cmap = plt.get_cmap(years_cmap)
            norm = plt.Normalize(min(multiyear), max(multiyear))
    if isinstance(normalize,str):
        df=normalize_df(df,normalize,columns=columns_to_plot)
    

    # actually plotting
    for i, col in enumerate(columns_to_plot):
        if xaxis == "index":
            df_nonan = df[[col]].dropna()
            df_nonan['Year'] = df_nonan.index.year
            df_nonan['xaxis'] = df_nonan.index  # Use the index as the x-axis
            xlim = [pd.to_datetime(limit, format='%Y/%m/%d') for limit in xlim]
            xaxis_name = 'Date'
        else:
            df_nonan = df[[col, xaxis]].dropna()
            df_nonan['Year'] = df_nonan.index.year
            df_nonan['xaxis'] = df_nonan[xaxis] 

        average = df_nonan.groupby('xaxis')[col].mean()
        std = df_nonan.groupby('xaxis')[col].std()

        if plot_together:
            color = colors[i]  # Use a unique color for each column
            if compare_years_to_baseline:
                for year in multiyear:
                    if year in df_nonan['Year'].unique():
                        year_data = df_nonan[df_nonan['Year'] == year]
                        year_data = year_data.sort_values(by='xaxis')
                        ax.plot(year_data['xaxis'], year_data[col], label=f'{col} {year}', color=cmap(norm(year)),linewidth=years_line_width)
                    else:
                        print(f"No {col} data available for year {year}")
            if plot_mean_std:
                ax.plot(average.index, average, color=color, label=f'mean {col} ±{k} std', alpha=1, linewidth=3)  # Mean line with full opacity
                ax.fill_between(average.index, average + k * std, average - k * std, color=color, alpha=std_alpha)  
            if plot_mean_std != 'only':
                df_nonan[col]=normalize_df(df_nonan[col],normalize)
                ax.scatter(df_nonan['xaxis'], df_nonan[col], marker='.', label=col, color=color, alpha=scatter_alpha)
            ax.set_xlabel(f'{xaxis_name}')
            ax.set_title(f'{Title}')
            ax.set_ylabel(f'{y_label}')
        else:
            # Individual plots for each column
            if compare_years_to_baseline:
                for year in multiyear:
                    if year in df_nonan['Year'].unique():
                        year_data = df_nonan[df_nonan['Year'] == year]
                        year_data = year_data.sort_values(by='xaxis')
                        ax[i].plot(year_data['xaxis'], year_data[col], color=cmap(norm(year)),linewidth=years_line_width)
                    else:
                        print(f"No {col} data available for year {year}")
        
            if plot_mean_std:
                ax[i].plot(average.index, average, color=colors[i], label=f'mean  {col} ±{k} std', alpha=1, linewidth=3)  
                ax[i].fill_between(average.index, average + k * std, average - k * std, color=colors[i], alpha=std_alpha)  
            if plot_mean_std != 'only':
                ax[i].scatter(df_nonan['xaxis'], df_nonan[col], marker='.', label=col, color=colors[i], alpha=scatter_alpha)
            if (xaxis=='Days since start of year') and (plot_break_up_dates): 
                #df_break_up.apply(lambda row: plt.text(row.index.dayofyear,row[col,row.name.index.year.strftime('YYYY')]))
              
                    
                ax[i].scatter(df_break_up.index.dayofyear, df_break_up[col], color='red',s=100,edgecolor='black', alpha=1)
                for dayofyear,break_up_date in zip(df_break_up.index.dayofyear,df_break_up.index):
                    ax[i].annotate(break_up_date.strftime('%Y'),
                                   (dayofyear,df_break_up.loc[break_up_date,col]),
                                   textcoords='offset points',
                                   xytext=(0,10),
                                   ha='left',
                                   rotation=10)            
            ax[i].set_ylabel(f'{col}')
            ax[i].set_title(f'{col}')
            ax[i].set_xlabel(f'{xaxis_name}')
            ax[i].set_xlim(xlim)
            ax[i].set_ylim(ylim[i] if ylim else None)

        if compare_years_to_baseline:
                  #
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax if plot_together else ax[i])
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if len(multiyear) == 1:
                single_year = multiyear[0]
                cbar.set_label('Year: {}'.format(single_year)) 
            else:
                  cbar.set_label('Year')
        if  compare_years_to_baseline:
            ax[i].legend()
        if  plot_together:
            ax.legend()   
   

    plt.tight_layout()
    plt.show()


def compute_and_plot_psd(df, cols=None, nperseg=None, plot_period=False, apply_filter=False, max_allowed_freq=None,
                         filter_order=4, find_peaks_kwargs=None,detrend_method='linear'):
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
            f, Pxx = welch(valid_data, fs=1.0, nperseg=nperseg if nperseg else len(valid_data)//2,detrend=detrend_method)
            
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
    Calculate the number of days since the last occurrence of a given date (MM/DD)  or a list of dates.

    Parameters
    -----------
    df: DataFrame
        The DataFrame containing the dates.
    date_or_dates: str or list of str
        A single date in the format `'MM/DD'`, a built-in keyword/string, or a list of dates in the format 'YYYY/MM/DD'.
    name: str, optional
        The name of the column to add when using a single date. If None, defaults to the month_day or special date keyword.

    Returns
    -------
    df: DataFrame
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


def plot_interactive_map(Pfafstetter_levels=4,plot_only_nearby_basin=True)-> px.choropleth_mapbox:
    """
        Plots an interactive map using Plotly Express

        Parameters
        ----------

        Pfafstetter_levels:  int
            The Pfafstetter level indicated the how detailed the basin is. Should be between 0 and 12, with level 0 correposninf to continent-spanning basins 
            and level 12 to very local basin

        plot_only_nearby_basin: bool
            If True, only plot basins near the nenana tripod. If False, plot all basins in the Arctic Region.

        Returns
        ----------

        plotly.express map with  markers in nenana tripod and  weather stations, polygon outline of the coverage of satellite-measured variables, and multipolygon with the outline of the basins.
    """
    if Pfafstetter_levels > 4 and plot_only_nearby_basin==False: 
        #gdf_temp = gpd.read_file(file, rows=1)  # Just read the first row to initialize and check length without laoding the whole file
        #gdf_basin_len = gpd.read_file(file).shape[0] # too slow withouf using external libries

        warnings.warn(f'Performance warning: Ploting basin at this level of detail could be slow', UserWarning)
        
        confirmation = input("Do you want to continue? (yes/no): ").strip().lower()
        if confirmation != 'yes':
            print("Operation cancelled.")
            return None 

    # plot near bases is Flase it lpots all the basin in the arctic region, if the pfastetter elvel is over 4 and plot_only_near_basin is False it coudl taka minute to create HTML interative plot
        
    # Define click event handler (to move around the map and get the coordinates)
    def click_callback(trace, points, selector):
        if points:
            lat = points.xs[0]
            lon = points.ys[0]
            print(f"Latitude: {lat}, Longitude: {lon}")

    plotly.offline.init_notebook_mode()


    # Latitude and longitude coordinates for weather station and other polygons manually added
    nenana_lat = 64.56702898502982
    nenana_lon = -149.0815700675435

    USW00026435_NENANA_LAT = 64.54725
    USW00026435_NENANA_LOG = -149.08713

    USW00026435_Fairbanks_LAT = 64.80309
    USW00026435_Fairbanks_LOG = -147.87606

    square_lat = [64, 64, 65, 65, 64]  # Latitude of vertices
    square_lon = [-150, -149, -149, -150, -150]  # Longitude of vertices

    gulkana_lat = 63.2818
    gulkana_lon = -145.426 

    usgs_tenana_river_lat = 64.5649444
    usgs_tenana_river_lon = -149.094 

    usgs_tenana_fairbanks_lat = 64.792344 
    usgs_tenana_fairbanks_lon = -147.8413097 

    square_lat_w = [64, 64, 66, 66, 64]  # Latitude of vertices
    square_lon_w = [-151, -149, -149, -151, -151]  # Longitude of vertices


 
    # changing the level to higher number yield more basin, using Pfafstetter levels 1-12 source HydroBASINS
    file='../../data/shape_files/hybas_lake_ar_lev'+'{:02d}'.format(Pfafstetter_levels)+'_v1c.shp'
    gdf_basin_lev = gpd.read_file(file)
    if plot_only_nearby_basin:
        if Pfafstetter_levels==1:
            gdf_basin_lev = gdf_basin_lev.iloc[[0]] # Filter the GeoDataFrame to include some basin ( its to heavy/slow if we include eveythin)
        elif Pfafstetter_levels==2:
            gdf_basin_lev = gdf_basin_lev.iloc[[0]]
        elif Pfafstetter_levels==3:
            gdf_basin_lev = gdf_basin_lev.iloc[[1]]
        elif Pfafstetter_levels==4:
            gdf_basin_lev = gdf_basin_lev.iloc[[15]]
        elif Pfafstetter_levels==5:
            gdf_basin_lev = gdf_basin_lev.iloc[[41,42,43,44]]
        elif Pfafstetter_levels==6:
            gdf_basin_lev = gdf_basin_lev.iloc[[80,81,82,82]]


   

    fig = px.choropleth_mapbox(
        gdf_basin_lev,
        geojson=gdf_basin_lev.geometry,
        locations=gdf_basin_lev.index,
        color=gdf_basin_lev.index,
        center={"lat": nenana_lat, "lon": nenana_lon},
        opacity=0.2,
        hover_name=gdf_basin_lev['HYBAS_ID'],
    )
    fig.update_layout(coloraxis_showscale=False)
  
 
    fig.update_traces(
    hovertemplate='<b>HydroBasin ID</b>: %{customdata}<extra></extra>',  # Custom hover text
    customdata=gdf_basin_lev['HYBAS_ID']  # Assign the data for hover text
)
 
    # ##########################################################33
    # gdf_rivers = gpd.read_file('../../data/shape_files/river_simplified_file.shp')
    # filtered_rivers = gdf_rivers[gdf_rivers['ORD_FLOW'] < 4]
    # print(len(filtered_rivers))
    # # Load the river shapefile using GeoPandas
 

    # river_coords = []
    # for geom in filtered_rivers.geometry:
    #     if geom.geom_type == 'LineString':
    #         river_coords.append(np.array(geom.coords))
    #     elif geom.geom_type == 'MultiLineString':
    #         for line in geom:
    #             river_coords.append(np.array(line.coords))

    # # Plot each river line on the map without adding to the legend
    # for coords in river_coords:
    #     latitudes, longitudes = coords[:, 1], coords[:, 0]  # Split into lat/lon
    #     fig.add_trace(go.Scattermapbox(
    #         lat=latitudes,
    #         lon=longitudes,
    #         mode='lines',
    #         line=dict(width=2, color='blue'),
    #         showlegend=False  # Do not show in legend
    #     ))

    fig.add_trace(go.Scattermapbox(
        lat=[nenana_lat], lon=[nenana_lon],
        mode='markers',
        marker=dict(size=10, color='purple', opacity=0.8),
        text=["NENANA tripod"],  # Text label for the marker
        hoverinfo="text",
        name="Ice classic tripod"))  # text on legend

    fig.add_trace(go.Scattermapbox(
        lat=[USW00026435_NENANA_LAT], lon=[USW00026435_NENANA_LOG],
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.8),
        text=["USGS Weather Station USW00026435"],
        hoverinfo="text",
        name="Nenana Weather Station"))

    fig.add_trace(go.Scattermapbox(
        lat=[USW00026435_Fairbanks_LAT], lon=[USW00026435_Fairbanks_LOG],
        mode='markers',
        marker=dict(size=10, color='red', opacity=0.8),
        text=["USGS Weather Station USW00026411"],
        hoverinfo="text",
        name="Fairbanks Weather Station"))

    fig.add_trace(go.Scattermapbox(
        lat=square_lat, lon=square_lon,
        mode='lines',  # Draw lines between vertices
        line=dict(color='yellow'),  # Color of the lines
        fill='toself',  # Fill the inside of the polygon
        fillcolor='rgba(255, 239,0, 0.1)',
        name='Temperature',
        text="Berkeley Earth Global",
        hoverinfo='text'))

    fig.add_trace(go.Scattermapbox(
        lat=[gulkana_lat], lon=[gulkana_lon],
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.8),
        name='Gulkana Glacier',
        text="USGS Weather Station 15485500",
        hoverinfo='text'))

    fig.add_trace(go.Scattermapbox(
        lat=[usgs_tenana_river_lat], lon=[usgs_tenana_river_lon],
        mode='markers',
        marker=dict(size=10, color='green', opacity=0.8),
        name='Tenana R at Nenana',
        text="USGS Weather Station 15515500",
        hoverinfo='text'))

    fig.add_trace(go.Scattermapbox(
        lat=[usgs_tenana_fairbanks_lat], lon=[usgs_tenana_fairbanks_lon],
        mode='markers', 
        marker=dict(size=10, color='green', opacity=0.8),
        name='Tenana R at Fairbanks',
        text="USGS Weather Station 15515500",
        hoverinfo='text'))

    fig.add_trace(go.Scattermapbox(
        lat=square_lat_w, lon=square_lon_w,
        mode='lines',
        line=dict(color='pink'), 
        fill='toself',  # Fill the inside of the polygon
        fillcolor='rgba(255, 20,147, 0.01)',
        name='Solar Radiation and Cloud Coverage',
        text="TEMIS & NERC-EDS",
        hoverinfo='text'))


    visibility_list_all=[True,True,True,True,True,True,True,True,True]
    visibility_list_weathers_stations=[False,True,True,True,False,True,True,True,False]
    visibility_list_basins=[True,False,False,False,False,False,False,False,False]

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=nenana_lat, lon=nenana_lon), zoom=5),
        margin=dict(l=0, r=0, t=0, b=0),  # Set margins to zero
        legend=dict(x=0, y=1, xanchor='left', yanchor='top', orientation='v'),
        updatemenus=[  # Add buttons for toggling layers
            dict(
                buttons=list([
                    dict(args=['visible', visibility_list_all],
                        label='Show all',
                        method='restyle'),
                    dict(args=['visible', visibility_list_weathers_stations],
                        label='Show weather Stations',
                        method='restyle'),
                    dict(args=['visible', visibility_list_basins],
                        label='Show basins'+str(Pfafstetter_levels),
                        method='restyle')
                ]),
                direction='left', pad={'r': 10, 't': 10}, showactive=True, type='buttons', x=0.1, xanchor='left', y=1.1, yanchor='top'),
        ])

    # Add invisible scatter plot trace to capture click events
    click_trace = go.Scattermapbox(lat=[], lon=[], mode='markers', marker=dict(opacity=0))
    fig.add_trace(click_trace)

    # Update click event handler
    click_trace.on_click(click_callback)

    # Show interactive plot
    fig.show()
    #fig.write_html('interactive_map.html')


def decimal_time(t, direction='to_decimal'):
    """ Convert time object to decimal and decimal to time object depending on the direction given

    Arguments:
        t : datetime object if `direction is 'to_decimal'`
            float if `direction='to_sexadecimal'`
    Returns:
        float if direction is 'to_decimal'
        datetime object if direction is 'to_sexadecimal'
    """

    if direction =='to_decimal':
        return t.hour+t.minute/60
    elif direction=='to_sexadecimal':
        hours=int(t)
        minutes=int((t-hours)*60)
        return time(hours,minutes)
    else:
        raise ValueError("Invalid direction, choose 'to_decimal'or 'to_sexadecimal'")
    

class IceModel(object):

    """
    Simple model that fits trend to historic data to extrapolate future break up date.

    Only considers previous break up dates. 


    The model compute the date and day separately

    METHODS:
        polyfit: Polinomic fit
        distfit: Fits distributions
        predict: Predicts the value of a variable based on the fit/dist
        get_prediction: Calles predict to get date and time
    """

    def __init__(self, df_dates:pd.DataFrame, df_variables=None):
        """
        Initializing object with DataFrame with break up dates.
        
        Arguments:
        ----------

        df_dates: pandas DataFrame
            DataFrame with break up dates.
        df_variables:pandas DataFrame, optional):
            DataFrame with additional variables. Defaults to None.
        """
        self._df_dates = df_dates.copy()
        self.df_variables = df_variables.copy() if df_variables is not None else pd.DataFrame()
        self._predicted_day_of_break_up = None  
        self._predicted_time_of_break_up = None
        
        # Initialize created properties tracker
        self._created_properties = set()

        # Dynamically create properties for each column in df_variables
        if not self.df_variables.empty:
            for column in self.df_variables.columns:
                self._create_property(column, self.df_variables[column])
    
    def _create_property(self, name, data):
        """Creates a property with a getter and setter for a given data Series or DataFrame column.
        
        Args:
            name (str): Name of the property.
            data (pandas Series): Data to be used for the property.
        """
        if len(data) != len(self._df_dates):
            raise ValueError("The length of the data must match the length of df_dates.")
        
        private_name = '_' + name

        setattr(self, private_name, data)
        
        def getter(self):
            return getattr(self, private_name)
        
        def setter(self, value):
            if len(value) != len(self._df_dates):
                raise ValueError("The length of the value must match the length of df_dates.")
            setattr(self, private_name, value)
        
        setattr(self.__class__, name, property(fget=getter, fset=setter))
        self._created_properties.add(name)

    def add_property(self, series, name_prop='new_property'):
        """Adds a new property to the class based on the provided Series.
        
        Args:
            series (pandas Series): Series to be added as a property.
            name_prop (str): Name of the new property.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("The argument must be a pandas Series.")
        if len(series) != len(self._df_dates):
            raise ValueError("The length of the series must match the length of df_dates.")
        if name_prop in self._created_properties:
            raise ValueError(f"Property '{name_prop}' already exists.")
        
        self._create_property(name_prop, series)
    
    def get_created_properties(self):
        """Returns a list of the properties dynamically created for the class."""
        return list(self._created_properties)

#=======================================================================================================
# Properties and methods related to df_dates 
#======================================================================================================
# DF without datetime index, each column correspond to break up dates in different format
# we can add more columns to this df but is important to understand that this df constaining one value per year
    
    # ------------------------------#
    # Properties basics
    # ------------------------------#
    @property
    def date_time(self):
        return  pd.to_datetime(self._df_dates[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
    @property
    def time(self):
        return self.date_time.dt.time
    @property
    def decimal_time(self):
        return self.time.apply(lambda t: decimal_time(t,direction='to_decimal'))
   
    @property
    def day_of_year(self):
        return self.date_time.dt.dayofyear.tolist()
   
    @property
    def year(self):
        return self.date_time.dt.year
     
    # ------------------------------#
    # Properties associated with Fits
    # ------------------------------#
    @property
    def fit_time(self):
        return self.fit_time
    
    @fit_time.setter
    def fit_time(self,value):   # revisar con test
        self.fit_time=value
    
    @property
    def fit_day_of_year(self):
        return self.fit_day_of_year
    @fit_day_of_year.setter
    def fit_day_of_year(self,value):
        self.fit_day_of_year=value
   
 
    # ------------------------------#
   # Properties to get predicted values
    # ------------------------------#
    @property
    def predicted_day_of_break_up(self,year=None):
        if self.predicted_day_of_break_up is None:
            raise ValueError(" Predicton of day of break up has not been made")
        return self.get_predicted_day(year)
    
    @predicted_day_of_break_up.setter
    def predicted_day_of_break_up(self,value):
        self._predicted_day_of_break_up=value
    
    @property
    def predicted_time_of_break_up(self):
        if self.predicted_time_of_break_up is None:
            raise ValueError(" Predicton of time of break up has not been made")
        return self.get_predicted_time
    @predicted_time_of_break_up.setter
    def predicted_dtime_of_break_up(self,value):
        self._predicted_time_of_break_up=value
   
    @property
    def prediction(self):
        if self._prediction is None:
            raise ValueError(" Predicton of  date and time of break up has not been made")
        return self.get_prediction
    @prediction.setter
    def prediction(self,value):
        self._prediction=value

   
     
    # ------------------------------#       
    # methods
    # ------------------------------#
    def polyfit(self,x_property: str ,y_property: str,degree: int=1,norm_order: int=2,print_eq:bool=True,plot:bool=False):
        """ 
        Fits polynomial function to properties of object

        The name of the property to use as 'x'  and  'y' must exist and be of the same length. The fit is saved as an attribute of the class 

        Parameters
        ---------- 
        x_property: str
            name of the property to use as 'x' in fit.
        y_property : str
             name of property to use as 'y' in fit
        degree : int
            degree of polynomial
        norm: int 
             degree of norm used to compute residuals, Default=2 
        print_eq: bool
            Determines if the equation of the fitted polynomial is printed
        plot: bool
            Determines if the plot of the data and the fitted polynomial is shown
    
        Returns
        ----------
        dict : dictionary with the fitted polynomial, name of the variables use for the fit and goodness of fit metrics. 
        """


        x=getattr(self,x_property)
        if y_property =='time':  # we want to use decimal time for the fit
            y_property='decimal_time'
        y=getattr(self,y_property)

        #print(x,y)

        coefs=np.polyfit(x,y,degree)

        
        polynomial=np.poly1d(coefs)
        
        if print_eq:
            print(polynomial)
            
        # G0odness of fit
        y_predict=polynomial(x)
        residuals=y-y_predict
        norm=np.linalg.norm(residuals,norm_order)  

        # these metrics are not generalized for higher order norms, they simply are the traditional metrics
        ss_res=np.sum(residuals**2)
        ss_tot=np.sum((y-np.mean(y))**2)

        r2=1-(ss_res/ss_tot)

        rmse=np.sqrt(np.mean((y-y_predict)**2))
        nrmse=rmse/(np.max(y)-np.min(y))

        n=len(y)  # number of points
        k=degree # how many coef are we estimating
        R2=1-((1-r2)*(n-1))/(n-k-1)
        
        gofs={f'{norm_order:}th norm':round(norm,4),'r2':round(r2,4),'R2':round(R2,4),'RMSE':round(rmse,4),'normalized RMSE':round(nrmse,4)}

        setattr(IceModel,'fit_'+str(y_property),{'Poly fit coefficients':polynomial,'(x,y)=':[x_property,y_property],'gofs metrics':gofs})

        if plot:
            plt.scatter(x, y, color='blue',alpha=0.5)
            x_ = np.linspace(min(x), max(x), 100)
            y_ = polynomial(x_)
        
            plt.plot(x_, y_, color='red', linestyle='--',linewidth=2)

            plt.xlabel(x_property)
            plt.ylabel(y_property)
            plt.title(f'Polynomial Fit degree={degree}')
            plt.grid(True)
            plt.show()

       # return {'Poly fit coefficients':polynomial,'(x,y)=':[x_property,y_property],'gofs metrics':gofs}
    
    def predict(self, variable: str, new_x) -> dict:
        """
        Uses the polynomic fit or distribution fit  associated with the y_property to predict y based on new value of x.

        Parameters
        ----------
        Variable: str
            Name of the property to check (e.g., 'decimal_time' for polynomial fit, etc.)
        new_x: float
            Value of x use to predict y.
        
        Returns
        ----------
        dict: A dictionary with information about the prediction:
            - (x,y): Tuple of the x and y properties used for fitting.
            - x_hat: The new x value used for prediction.
            - y_hat: The predicted y value.
            - confidence_interval: The confidence interval for the prediction (only for distributions).
        """

        if not self.check_property(variable):
            raise AttributeError(f"Variable '{variable}' is not part of the predicted variables")
        
        fit = getattr(self, 'fit_' + str(variable))
    
        if 'Poly fit coefficients' in fit:
            # Polynomial fit
            fit_coefs = fit['Poly fit coefficients']
            predicted_y = fit_coefs(new_x)
            return {
                '(x,y)': fit['(x,y)='],
                'x_hat': new_x,
                'y_hat': round(predicted_y, 4)
            }
        
        elif 'Fitted Distribution' in fit:
            # Distribution fit
            distribution = fit['Fitted Distribution']
            params = fit['Parameters']
            dist = getattr(stats, distribution)

            # Compute the predicted value (expected value of distribution)
            predicted_y = dist(*params).mean()

            # Confidence interval
            if distribution == 'norm':
                ci =  1.96* dist(*params).std() 
                lower_bound = predicted_y - ci
                upper_bound = predicted_y + ci
                confidence_interval = (round(lower_bound, 4), round(upper_bound, 4))
            else:
                confidence_interval = 'N/A'  # finish this

            return {
                '(x,y)': fit['(x,y)='],
                'x_hat': new_x,
                'y_hat': round(predicted_y, 4)
            }
        
        else:
            raise AttributeError(f"No fit found for variable '{variable}'")
    
    def check_property(self,prop_name):
        """
        simple method that check if a fit corresponding to that variable exists
        """
        if not hasattr(self,prop_name):
            raise AttributeError(f'variable "{prop_name}" not part of the model')
        else: 
            return True
        
    def get_prediction(self,x_vars)->datetime:
        """
         Calls the function to predict the date (day of year) and time of break up
         

        Parameters
        ----------
        xvars: list
            List with the x variables that will be use to predict with date and time (respectably).

        Notes
        ----------
            The method is not generalized for other properties, it is only for date and time. For predicting other properties,
                it is necessary to use .predict() which can receive any two existing properties.
            A list of x_vars is required instead of a single value, as in other future complex model, the predicted date/time could correspond 
                to a combination of variables for different years.

        Returns
        ----------
        datetime: The predicted date and time of break up
        """
    
            # we are re-getting  the value just to make sure they correspond to the latest assigned fits

        #DATE
        x_fit_date=x_vars[0]
        day=self.predict('day_of_year', x_fit_date)
        date=datetime(x_fit_date,1,1)+timedelta(days=int(day['y_hat'])-1)
        
        #TIME
        x_fit_time=x_vars[1]
        time=decimal_time(self.predict('decimal_time',x_fit_time)['y_hat'],direction='to_hexadecimal')
        #Combine
        self._prediction=datetime.combine(date,time)
        print(self._prediction)

    def dist_fit(self, x_property: str, y_property: str, distribution: str = 'norm', print_eq: bool = True,ci=1.96,plot=False): 
        """ Fit a distribution to properties of the object

        Args:
            x_property (str): Name of the x property (not used in this implementation but kept for consistency)
            y_property (str): Name of the y property
            distribution (str): Name of the distribution to fit (from scipy.stats). Default is 'norm'.
                                'norm' for normal distribution
                                'expon' for exponential distribution
                                'gamma' for gamma distribution
                                'lognorm' for lognormal distribution
                                'weibull_min' for Weibull distribution
                                'weibull_max' for Frechet distribution
                                'pareto' for Pareto distribution
                                'genextreme' for Generalized Extreme Value distribution
                                'gumbel_r' for Gumbel Right (minimum) distributionE
                                ...
            plot (bool): Determines if a histogram of the data and the fitted distribution are plotted. Default is False.   
            print_eq (bool): Determines if the equation of the fitted distribution is printed. Default is True.
            ci (float): Confidence interval for the prediction. Default is 1.96 (95% CI). generalize this for other distributions

        Prints:
            Parameters of the fitted distribution and goodness-of-fit metrics.

        Returns:
            dict: Dictionary with fitted distribution, names of the variables used for the fit, and goodness-of-fit metrics.
        """

        
        if not hasattr(self, x_property):
            print(f"Property '{x_property}' not found .")
            return {}

        if y_property == 'time':  # Convert 'time' to 'decimal_time' if needed
            y_property = 'decimal_time'

        if not hasattr(self, y_property):
            print(f"Property '{y_property}' not found in the object.")
            return {}

        x = getattr(self, x_property)
        y = getattr(self, y_property)


        # Check if the distribution is valid (could take out as itis mention in description)
        if not hasattr(stats, distribution):
            print(f"Distribution '{distribution}' not found in scipy.stats.")
            return {}

        dist = getattr(stats, distribution)
        
        # Fit the distribution
        params = dist.fit(y)
        fitted_dist = dist(*params)

        # Goodness-of-fit metrics
        ks_stat, ks_p_value = stats.kstest(y, fitted_dist.cdf)
        
        gofs = {
            'KS Statistic': round(ks_stat, 4),
            'KS p-value': round(ks_p_value, 4),
        }

        if print_eq:
            print(f"Distribution: {distribution}")
            print(f"Parameters: {params}")

        results = {
            'Fitted Distribution': distribution,
            'Parameters': np.round(params,4),
            '(x,y)=': [x_property, y_property],
            'Goodness-of-Fit Metrics': gofs
        }
        if plot:
            # Histogram of the data
            plt.hist(y, density=True, alpha=0.6, color='g')

            # Plot the PDF of the fitted distribution
            xmin, xmax = min(y), max(y)
            x = np.linspace(xmin, xmax, 100)
            p = fitted_dist.pdf(x)
            plt.plot(x, p, 'k', linewidth=2, label=f'{distribution} fit')

            plt.xlabel(y_property)
            plt.ylabel('Density')
            plt.title(f'Histogram of {y_property} with Fitted {distribution} distribution')
            plt.legend()
            plt.grid(True)
            plt.show()

        setattr(self, 'fit_' + str(y_property), results)
#=======================================================================================================
# Properties related to df_dates 
#======================================================================================================
   # dynamically create when the object is created
#======================================================================================================
#======================================================================================================
#======================================================================================================
        return results
    
def plot_scatter(dates:pd.DataFrame, x_col_name:str,y_col_name:str,x_label: str=None,y_label:str=None,title:str=None):
    """_summary_

    Arguments
    ----------
    dates:pd.dataframe
        Datafrmae, each row correspond to a year with charactetistic of the break_up and associated variable. Index is datetie object with years
    x_col_name: str
        name of the column to use as x vector
    y_col_name: str
        name of the column to use as y vector
    x_label: None | str
        optional string to label xaxis
    ylabel: None | str
        optional string to label yaxis
    Title:  None | str
        title of scatter plot

    Returns
    -------
    plt.fig
    """

    if x_label is None:
        x_label=x_col_name
    if y_label is None:
        y_label=y_col_name

    x=dates[x_col_name]
    y=dates[y_col_name]
    plt.figure(figsize=(15,5))
    plt.scatter(x,y,color='red',s=50,edgecolor='black')
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title,pad=20)
    for i in range(len(dates)):
        # print(dates.index[i])
        #print(dates.iloc[i,x_col_name])
        plt.annotate(dates.index[i],
                        (dates[x_col_name].iloc[i],dates[y_col_name].iloc[i]),
                        textcoords='offset points',
                        xytext=(0,10),
                        ha='left',
                        rotation=10)
    plt.show()     

def decimal_time(t, direction='to_decimal'):
    """ Convert time object to decimal and decimal to time object depending on the direction given

    Arguments:
        t : datetime object if `direction is 'to_decimal'`
            float if `direction='to_hexadecimal'`
    Returns:
        float if direction is 'to_decimal'
        datetime object if direction is 'to_hexadecimal'
    """

    if direction =='to_decimal':
        return t.hour+t.minute/60
    elif direction=='to_sexadecimal':
        hours=int(t)
        minutes=int((t-hours)*60)
        return time(hours,minutes)
    else:
        raise ValueError("Invalid direction, choose 'to_decimal'or 'to_hexadecimal'")

def decimal_day(t, direction='to_decimal'):
    """ Convert time object to decimal and decimal to time object depending on the direction given

    Arguments:
        t : datetime object if `direction is 'to_decimal'`
            float if `direction='to_base24'`
    Returns:
        float if direction is 'to_decimal'
        datetime object if direction is 'to_base24'
    """

    if direction =='to_decimal':
        return t.hour/24+t.minute/(24*60)
    elif direction=='to_base24':
        total_minutes=int(t*24*60)
        hours=minutes//60
        minutes=total_minutes%60
        return time(hours,minutes)
    else:
        raise ValueError("Invalid direction, choose 'to_decimal'or 'to_base24'")
    

    