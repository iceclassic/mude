import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
                    norm_type: str | None = None,
                    correlation: bool = False
                    ) -> None:
    """
    Function that print the contents fo the dataframe a plot the content/distribution of each column

    :param data: Pandas dataframe object, where the index is a datetime object
    :param cols: list of column names as strings to compare
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
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
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

