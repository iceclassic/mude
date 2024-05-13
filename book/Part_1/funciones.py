import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explore_contents(Data,colormap='viridis',opt={'info':True,'Time History': True,'Sparcity':True}):
    '''
    Function that print the contents fo the dataframe a plot the content/distribution of each column
    +++++++++++++++
    Data: Pandas dataframe, where the index is a datetime object 
    colormap: name of matplotlib cmaps
    opt: Dictionary with options of different way to explore the contents of the df
        info: uses built_in methods of pandas to get column, dtype, number of entries and range of entries as basic column statistics
        Time History: Plots the contents of every column and the distribution of the values on it
        Sparsity: Heatmap of contents, compares the 'frequency sampling' of each colum  
    '''
    Data=Data.copy()
    if opt['info']:
        Data.info()

    if opt['Time History']:
        fig, axs = plt.subplots(nrows=len(Data.columns), ncols=2, figsize=(20, 3*len(Data.columns)), 
                                gridspec_kw={'width_ratios': [3, 1]})  # Adjust the width ratio here

        plt.subplots_adjust(wspace=0.2)  

        for i, col in enumerate(Data.columns):
            # Plot line 
            col_data = Data[col].copy()
            col_data.dropna(inplace=True)
            if not col_data.empty:
                axs[i, 0].plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
                axs[i, 0].legend()
                axs[i, 0].set_title(str(col)+': Time Series')  # Title for the line plot
            # Plot density 
                Data[col].plot.density(ax=axs[i, 1])
                axs[i, 1].set_xlim(left=Data[col].min(), right=Data[col].max())  # Set x-axis limits to column range
                axs[i, 1].set_ylabel('Density')
                axs[i, 1].set_title(str(col)+': Distribution')  # Title for the line plot
        plt.tight_layout()
        plt.show()
    if opt['Sparcity']:
        Data.index = Data.index.year
        plt.figure(figsize=(20, 10))
        sns.heatmap(Data.T.isnull(), cbar=False, cmap=colormap, yticklabels=Data.columns)
        plt.title('Sparsity of Time-Series')
        plt.show()

def compare_columns(Data,cols,norm_type='none',correlation=False):
    '''
    Function that print the contents fo the dataframe a plot the content/distribution of each column
    +++++++++++++++
    Data: Pandas dataframe, where the index is a datetime object 
    col: list of column to compare
    normtype
    '''
    Data=Data.copy()
    Data_selected=Data[cols]
    fig, axs = plt.subplots(figsize=(20, 6))

    plt.subplots_adjust(wspace=0.2)  
    Data_selected=normalize_df(Data_selected,norm_type)
    for i, col in enumerate(Data_selected.columns):
            # Plot line 
            col_data = Data_selected[col].copy()
            col_data.dropna(inplace=True)
            if not col_data.empty:
                axs.plot(col_data.index, col_data.values, label=col, color=plt.cm.tab10(i % 10))
                axs.legend()
                axs.set_title('Time Series')  # Title for the line plot
    plt.tight_layout()
    plt.show()
#
    if correlation:
        # Compute correlation considering data only where both columns have data
        correlation_matrix = Data_selected.dropna().corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()  # Because the columns have different sampling frequencies, the correlation computed wrong. Compute the correlation considering data only where both columns have data.
   

def normalize_df(df,norm_type):
    '''
    We are avoiding using sklearn to be light
    '''
    df_normalized = df.copy()  # Make a copy of the DataFrame
    if norm_type=='none':
        return df_normalized
    elif norm_type == 'min_max':
        for column in df.columns:
            #eeee
            df_normalized[column] = min_max_normalization(df[column])
    elif norm_type == 'z-norm':
        for column in df.columns:
            df_normalized[column] = z_score_normalization(df[column])
    else:
        raise ValueError("Invalid inout in normalization type")


def min_max_normalization(column):
    '''
    light min_max scalling implementaion
    
    '''
    column_numeric = pd.to_numeric(column, errors='coerce')
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)
    return scaled_column

def z_score_normalization(column):
    '''
    basic z norm
    
    '''
    column= pd.to_numeric(column, errors='coerce')
    mean = column.mean()
    std_dev = column.std()
    normalized_column = (column - mean) / std_dev
    return normalized_column

