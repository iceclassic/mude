import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

def explore_contents(data: pd.DataFrame,
                     colormap: str = 'viridis',
                     opt: dict = {'Info':True,
                                  'Time History': True,
                                  'Sparsity':True}
                     ) -> None:
    """
    Function that print the contents fo the dataframe a plot the content/distribution of each column
    
    :param data: Pandas DataFrame object, where the index is a datetime object 
    :param colormap: Name of the matplotlib cmap to use
    :param opt: Dictionary with options of different way to explore the contents of the df
        Info: uses built_in methods of pandas to get column, dtype, number of entries and range of entries as basic column statistics
        Time History: Plots the contents of every column and the distribution of the values on it
        Sparsity: Heatmap of contents, compares the 'frequency sampling' of each colum  
    :return: None
    """

    # Make a copy of the input data
    data = data.copy()

    if opt['Info']:
        data.info()

    if opt['Time History']:
        fig, axs = plt.subplots(nrows=len(data.columns), ncols=2, figsize=(20, 3*len(data.columns)), 
                                gridspec_kw={'width_ratios': [3, 1]})  # Adjust the width ratio here
        plt.subplots_adjust(wspace=0.2)  

        for i, col in enumerate(data.columns):
            # Plot line 
            col_data = data[col].copy()
            col_data.dropna(inplace=True)
            if not col_data.empty:
                axs[i, 0].plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
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
                    colormap: str = 'magma',
                    norm_type: str | None = None,
                    correlation: bool = False
                    ) -> None:
    """
    Function that print the contents fo the dataframe a plot the content/distribution of each column
    :param data: Pandas dataframe object, where the index is a datetime object
    :param cols: list of column names as strings to compare
    :param colormap: Name of the matplotlib cmap to use
    :param norm_type:
    :param correlation:
    :return:
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
        # Compute correlation considering data only where both columns have data
        # !!Because the columns have different sampling frequencies, the correlation computed wrong.
        # Compute the correlation considering data only where both columns have data.!!
        correlation_matrix = data_selected.dropna().corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap=colormap, fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()


def normalize_df(df: pd.DataFrame,
                 norm_type: str | None = None
                 ) -> pd.DataFrame:
    """
    Normalizes the Pandas DataFrame object.
    :param df:
    :param norm_type:
    :return:
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
        raise ValueError(f"Invalid input for norm_type: {norm_type}. Please provide a valid normalization type.")

    return df_normalized


def min_max_normalization(column: pd.Series) -> pd.Series:
    """
    Normalizes a pandas DataFrame Series using lightweight min-max-normalization
    :param column: Column to normalize as a pandas.Series
    :return: The normalized column as a pandas.Series
    """

    column_numeric = pd.to_numeric(column, errors='coerce')
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)

    return scaled_column


def z_score_normalization(column: pd.Series) -> pd.Series:
    """
    Normalizes a pandas DataFrame Series using basic z-normalization.
    :param column: Column to normalize as a pandas.Series
    :return: The normalized column as a pandas.Series
    """

    column = pd.to_numeric(column, errors='coerce')
    mean = column.mean()
    std_dev = column.std()
    normalized_column = (column - mean) / std_dev

    return normalized_column


def filter_df(df,start_date=None,end_date=None, cols=None, multiyear=None):
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
    # Ensure multiyear is a list if not provided
    if multiyear is None:
        multiyear = []

    if multiyear:
        df = df[df.index.year.isin(multiyear)]

    # Filter by month/day range if both start_date and end_date are provided
    if (start_date is not None) and (end_date is not None):
        start_date = pd.to_datetime(start_date, format='%m/%d')
        end_date = pd.to_datetime(end_date, format='%m/%d')
        mask = (df.index.month == start_date.month) & (df.index.day >= start_date.day) \
| (df.index.month == end_date.month) & (df.index.day <= end_date.day)
        

        df = df[mask]

    # Select specific columns if provided
    if cols is not None:
        df = df[cols]

    return df


def plot_columns_interactive(df, column_groups, title=None, xlabel=None, ylabel=None, y_domains=None):
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
    ylabel : str, optional
        The label for the y-axis.
    y_domains : dict, optional
        A dictionary where keys are integers representing the y-axis index and values are lists of two floats representing the domain of the y-axis.
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
                title=f"{group_name} [{ylabel}]", 
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
        legend=dict(traceorder="reversed"),
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
                 "line": {"color": 'red', "width": 0.6, "dash": 'dot'}, 'name': 'break up time'}
        shapes.append(shape)

    fig.update_layout(shapes=shapes)

    fig.show()
