"""
author : jhhalls
copyrights reserved
"""


import streamlit as st
import pandas as pd
import io
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# cosmetic displays
from cosmetic_display import get_df_info
# Categorical PLots
from categorical_plots import bar_chart, pie_chart, donut_chart, pareto_chart, count_plot
from categorical_plots import frequency_table, waffle_chart, tree_map, heatmap, dot_plot, bubble_chart
# Continuous Plots
from numerical_plots import plot_histogram, plot_boxplot, plot_violin, density_plot, qq_plot, rug_plot
from numerical_plots import line_plot, scatter_plot, ecdf_plot, hexbin_plot, strip_plot, swarm_plot, frequency_distribution
# bivariate plots
from bivariate_plots import scatter_plot, line_plot, joint_plot, hexbin_plot, kde_plot, hist2d_plot, box_plot, correlation_heatmap
from bivariate_plots import regplot, swarm_plot, strip_plot, lollipop_plot, bubble_chart, step_plot
# multivariate plot
from multivariate_plots  import scatter_matrix_plot
# set the page layout to fit the width of the webpage
st.set_page_config(layout="wide")


st.write("All imports are successful.")
st.write("The environment is ready to upload your data")
st.write("""
         # No Code Machine Learning App
         """)

st.subheader("Upload the Raw Data (csv file)")
with st.spinner("Please wait..."):
    uploaded_file = st.file_uploader("Select a file")
                                     
df = pd.read_csv(uploaded_file, on_bad_lines=  "skip", encoding="utf-8")
st.markdown(""":rainbow[Data loaded Successfully]""")
st.write("---")
# ============================ Have a Look into the data============================
# upload data
st.subheader("Have a Glance")
# drop down for get a glance
selected_option_1 = st.selectbox(label = "Select a method to view data",
                                 placeholder= "Select one to have a look at the data",
                                 options = ["Head", "Tail", "Sample"],
                                 label_visibility="hidden")

# display output based on the selected option
if selected_option_1 == "Head":
    head = df.head()
    st.write(head)
elif selected_option_1 == "Tail":
    tail = df.tail()
    st.write(tail)
elif selected_option_1 == "Sample":
    sample = df.sample(3)
    st.write(sample)

st.write("Does your data has 'Unnamed: 0' as the first column")
col1,col2 = st.columns(2)
with col1:
    checkbox_1 = st.checkbox(label="Yes")
with col2:
    checkbox_2 = st.checkbox(label="No")

if checkbox_1:
    df = df.set_index(df.columns[0])
else:
    df = df

st.write("---")
#  ============================//Have a Look into the data============================


#  ============================Data Summary============================
st.header("Data Exploration", divider=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    summary_label_1 = "Shape"
    checkbox_1 = st.checkbox(summary_label_1, help="Displays the number of Rows and Columns in the Dataframe")
with col2:
    summary_label_2 = "#Rows"
    checkbox_2 = st.checkbox(summary_label_2, help="Displays the number of Rows in Dataframe")
with col3:
    summary_label_3 = "#Columns"
    checkbox_3 = st.checkbox(summary_label_3, help="Displays the number of columns in the Dataframe")
with col4:
    summary_label_4 = "Datatype"
    checkbox_4 = st.checkbox(summary_label_4, help="Displays Datatype of each of the individual Columns")
with col5:
    summary_label_5 = "Columns"
    checkbox_5 = st.checkbox(summary_label_5, help="Displays the list of Columns")

columns = df.columns.to_list()
if checkbox_1:
    selected_option = st.markdown("You selected the option : :green[{}]".format(summary_label_1) )
    shape = df.shape
    st.write("The shape of the DataFrame is ", shape)
elif checkbox_2:
    selected_option = st.write("You selected the option :", summary_label_2 )
    nrows = df.shape[0]
    st.write("Number of Rows in the DataFrame are ",nrows)
elif checkbox_3:
    selected_option = st.write("You selected the option :", summary_label_3)
    ncols = df.shape[1]
    st.write("Number of Columns in the dataframe are ", ncols)
elif checkbox_4:
    selected_option = st.write("You selected the option :", summary_label_4 )
    col_dtypes = df.dtypes
    st.write(col_dtypes)
elif checkbox_5:
    selected_option = st.write("You selected the option :", summary_label_5)
    st.write("The Columns in the DataFrame are : ",columns)

st.write("---")
#  ============================//Data Summary============================
    

#  ============================Data Exploration============================
selected_option_2 = st.selectbox(label = "Select a method to explore data",
                                placeholder= "Select one to have a look at the data",
                                options = ["Describe", "Info", "Null Values"])

# display output based on the selected option
if selected_option_2 == "Describe":
    col1, col2, col3 = st.columns(3)
    with col1:
        label_1 = "Continuous Cols"
        checkbox_describe_1 = st.checkbox(label=label_1, 
                                          help="These are the columns that contains real numbers that can be displayed on a Number Line")
    with col2:
        label_2 = "Categorical Cols"
        checkbox_describe_2 = st.checkbox(label=label_2, 
                                          help="Columns containing categories. Eg. Male/Female")
    with col3:
        label_3 = "All Cols"
        checkbox_describe_3 = st.checkbox(label=label_3, 
                                          help="All the columns are taken and both the previous checkkbox results are combined.")

    if checkbox_describe_1:
        selected_option = st.write("You selected the option :", label_1 )     # print the selected label
        describe = df.describe(include=[np.number]).T                         # count, mean, quartiles, min, max
        st.write(describe)                                                    # display describe
    elif checkbox_describe_2:
        try:
            selected_option = st.write("You selected the option :", label_2 )     # print the selected label
            describe = df.describe(include=['O']).T                               # count, #Unique, Mode, freq of mode
            st.write(describe)                                                        # display describe
        except:
            st.write("There are no Categorical Columns")
    elif checkbox_describe_3:
        selected_option = st.write("You selected the option :", label_3 )     # print the selected label
        describe = df.describe(include="all").T                               # 
        st.write(describe)
    # st.write(describe)
elif selected_option_2 == "Info":
    buffer = io.StringIO()
    df.info(buf=buffer)
    buffer_get_value = buffer.getvalue()
    get_df_info(buffer_get_value)
elif selected_option_2 == "Null Values":
    null_data = df.isna().sum()
    total_null = df.isna().sum().sum()
    st.write("Total Null values count is ", total_null)
    st.write("Null values per columns is as follow : \n", null_data)
    st.write("Would you like to Fill the Null Values : ")
    col1, col2 = st.columns(2)
    with col1:
        impute_label_1 = "Yes"
        checkbox_impute_1 = st.checkbox(label=impute_label_1)
    with col2:
        impute_label_2 = "No"
        checkbox_impute_2 = st.checkbox(label=impute_label_2)
    if checkbox_impute_1:
        st.write("You have selected : ", impute_label_1)
        st.write("Add methods to process Null values later")
        # logic here
    elif checkbox_impute_2:
        st.write("You have selected : ", impute_label_2)
        st.markdown(":red[Be assured that you cannot train ML Model with null values in Data]")
        # logic here
    else:
        st.markdown(":red[You have not selected any option for Imputation.\n Hence the Null values continue to exist.]")

st.write("---")
#  ============================//Data Exploration============================


#  ============================Handle Null values============================

#  ============================//Handle Null values============================


#  ============================Data Visualization============================
st.header("Data Visualization", divider=True)

select_no_of_variables = st.selectbox('Select one option:', ['', "Univariate Analysis - Analyse Single column", "Bi-Variate Analysis - Analyze Two Columns", "Multivariate Analysis - Analyze Multiple Columns"], 
                                   format_func=lambda x: 'Select an option' if x == '' else x)

if select_no_of_variables == "Univariate Analysis - Analyse Single column":
    st.markdown("You have Selected : :rainbow[ {} ]".format(select_no_of_variables))
    # univariate
    univariate_analysis = st.selectbox('Select one option:', ['', "Categorical  - Visualize Categorical Columns", "Numerical - Visualize Numerical Columns"], 
                                    format_func=lambda x: 'Select an option' if x == '' else x)

    # Univariate Categorical Analysis
    if univariate_analysis == "Categorical  - Visualize Categorical Columns":
        st.markdown("You have Selected : :rainbow[ {} ]".format(univariate_analysis))
        # st.write("You have Selected : ", univariate_analysis)

        # ==============Categorical Columns==============
        categorical_df = df.select_dtypes('O')
        categorical_cols = categorical_df.columns
        st.markdown("The Following are the :orange[Categorical Columns] in the current data: ")
        st.write(categorical_df.columns)
        
        st.write(":orange-background[Let's Begin with the  :blue[**Categorical Visualization**]]")
        col1, col2 = st.columns(2)
        with col1:
            select_categorical_col = st.selectbox(label="Select columns", 
                                                help="Select the columns you want to visualize",
                                                options=categorical_cols)
        
        with col2:
            select_cat_plots = st.selectbox("Select Chart Type", ["Frequency Table","Pie Chart", "Bar Chart", "Count Plot" ,"Area Chart", "Donut Chart", "Heatmap","Bubble Plot" ,"Dot Plot" ,"Pareto Chart", "Waffle Chart", "Tree Map"],
                                            format_func=lambda x: 'Select an option' if x == '' else x)
        
        # ======Categorical plots===========
        if select_cat_plots == "Frequency Table":
            graph = frequency_table(df=df, column=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Bar Chart":
            graph = bar_chart(df=df, column=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Count Plot":
            graph = count_plot(df=df, x_col=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Pie Chart":
            graph = pie_chart(df=df, column=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Donut Chart":
            graph = donut_chart(df=df, column=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Heatmap":
            graph = heatmap(df=df, column= select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Bubble Plot":
            graph = bubble_chart(df=df, column= select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Dot Plot":
            graph = dot_plot(df=df, column= select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Pareto Chart":
            graph = pareto_chart(df=df, column=select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Area Chart":
            graph = test_fig()
            st.write(graph)
        elif select_cat_plots == "Waffle Chart":
            graph = waffle_chart(df=df, column= select_categorical_col)
            st.write(graph)
        elif select_cat_plots == "Tree Map":
            graph = tree_map(df=df, column= select_categorical_col)
            st.write(graph)
        # ==================== Categorical Plots (ends)=========================

    # ==================== Continues Plots (ends)=========================
    # Univariate Continuous Analysis
    elif univariate_analysis == "Numerical - Visualize Numerical Columns":
        st.markdown("You have Selected : :green[ {} ]".format(univariate_analysis))

        # st.write("You have Selected : ", univariate_analysis)
        # select numeric data
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df_continuous = df.select_dtypes(include=numerics)
        df_continuous_columns = df_continuous.columns
        # Display numerical columns
        st.markdown("The Following are the :orange[Numerical Columns] in the current data: ")
        st.write(df_continuous_columns)
        st.write(":orange-background[Let's Begin :blue[**Numerical Visualization**]]")

        # Create two dropdowns.
        col1, col2 = st.columns(2)
        # 1st Dropdown - To select column
        with col1:
            select_numerical_col = st.selectbox(label="Select columns", 
                                                help="Select the columns you want to visualize",
                                                options=df_continuous_columns)
        # 2nd Dropdown - To select the type of graph
        with col2:
            select_numeric_plots = st.selectbox("Select Chart Type", ["Frequency Distribution Table", "Histogram","Density Plot","Scatter Plot","Box Plot","Violin Plot","Q-Q Plot","Rug Plot", "Line Plot" ,"ECDF Plot", "Hexbin Plot","Strip Plot","Swarm Plot","test_1"],
                                            format_func=lambda x: 'Select an option' if x == '' else x)
        
        # ======Numerical plots===========
        if select_numeric_plots == "Frequency Distribution Table":
            graph = frequency_distribution(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Histogram":
            graph = plot_histogram(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Density Plot":
            graph = density_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Scatter Plot":
            graph = scatter_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Box Plot":
            graph = plot_boxplot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Violin Plot":
            graph = plot_violin(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Q-Q Plot":
            graph = qq_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Rug Plot":
            graph = rug_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Line Plot":
            graph = line_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "ECDF Plot":
            graph = ecdf_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Hexbin Plot":
            graph = hexbin_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Strip Plot":
            graph = strip_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Swarm Plot":
            graph = swarm_plot(df = df, column = select_numerical_col)
            st.write(graph)
        elif select_numeric_plots == "Bar Chart":
            graph = bar_chart(df = df, column = select_numerical_col)
            st.write(graph)
        # ends here
elif select_no_of_variables == "Bi-Variate Analysis - Analyze Two Columns":
    st.markdown("You have Selected : :rainbow[ {} ]".format(select_no_of_variables))
    st.write("Work on Bi- Variate Analysis in this section")

    col1, col2 = st.columns(2)
    with col1:
        select_bivariate_col_1 = st.selectbox(label="Select First column", 
                                            help="Select the column you want to visualize",
                                            options=columns)
    
    with col2:
        select_bivariate_col_2 = st.selectbox(label="Select Second column", 
                                            help="Select the column you want to visualize",
                                            options=columns)
    
    if select_bivariate_col_1 == select_bivariate_col_2:
        if df[select_bivariate_col_1].dtype == "O":
            graph = count_plot(df=df, x_col=select_bivariate_col_1)
            st.write(graph)
        elif df[select_bivariate_col_1].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            graph = plot_histogram(df=df, column=select_bivariate_col_1)
            st.write(graph)
    else:
        select_bivariate_plot = st.selectbox('Select the Type of Plot:', ['', "Scatter Plot", "Line Plot","Joint Plot", "Hexbin Plot", "KDE Plot", "2D Histogram Plot", "Box Plot", "Correlation Heatmap", "Regression Plot", "Swarm Plot", "Strip Plot","Lollipop Plot", "Bubble Plot", "Step Plot"], 
                                   format_func=lambda x: 'Select an option' if x == '' else x)
        if select_bivariate_plot == "Scatter Plot":
            graph = scatter_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
            st.write(graph)
        elif select_bivariate_plot == "Line Plot":
            graph = line_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
            st.write(graph)
        elif select_bivariate_plot == "Joint Plot":
            select_joint_plot_kind = st.selectbox('Do you wish to tweak the plot type', ['', "scatter", "kde", "hist", "hex", "reg" , "resid"], 
                                   format_func=lambda x: 'Select an option' if x == '' else x)
            if select_joint_plot_kind == "scatter":
                graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                st.write(graph)
            elif select_joint_plot_kind == "kde":
                try:
                    graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                    st.write(graph)
                except:
                    st.markdown(":red[Both the columns should be Continuous in nature]")
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    df_continuous = df.select_dtypes(include=numerics)
                    st.write("The Continuous columns in the data are :", df_continuous.columns)
            elif select_joint_plot_kind == "hist":
                graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                st.write(graph)
            elif select_joint_plot_kind == "hex":
                try:
                    graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                    st.write(graph)
                except:
                    st.markdown(":red[Both the columns should be Continuous in nature]")
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    df_continuous = df.select_dtypes(include=numerics)
                    st.write("The Continuous columns in the data are :", df_continuous.columns)
            elif select_joint_plot_kind == "reg":
                try:
                    graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                    st.write(graph)
                except:
                    st.markdown(":red[Both the columns should be Continuous in nature]")
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    df_continuous = df.select_dtypes(include=numerics)
                    st.write("The Continuous columns in the data are :", df_continuous.columns)
            elif select_joint_plot_kind == "resid":
                try:
                    graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2, kind=select_joint_plot_kind)
                    st.write(graph)
                except:
                    st.markdown(":red[Both the columns should be Continuous in nature]")
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    df_continuous = df.select_dtypes(include=numerics)
                    st.write("The Continuous columns in the data are :", df_continuous.columns)
            else:
                graph = joint_plot(df=df, x_col=select_bivariate_col_1, y_col=select_bivariate_col_2)
                st.write(graph)
        elif select_bivariate_plot == "Hexbin Plot":
            try:
                graph = hexbin_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "KDE Plot":
            try:
                graph = kde_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "2D Histogram Plot":
            try:
                graph = hist2d_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "Box Plot":
            try:
                graph = box_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "Correlation Heatmap":
            try:
                graph = correlation_heatmap(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "Regression Plot":
            try:
                graph = regplot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
            except:
                st.markdown(":red[Both the columns should be Continuous in nature]")
                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                df_continuous = df.select_dtypes(include=numerics)
                st.write("The Continuous columns in the data are :", df_continuous.columns)
        elif select_bivariate_plot == "Swarm Plot":
                graph = swarm_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
        elif select_bivariate_plot == "Strip Plot":
                graph = strip_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
        elif select_bivariate_plot == "Lollipop Plot":
                graph = lollipop_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                st.write(graph)
        elif select_bivariate_plot == "Bubble Plot":
                y_col_bubble_size = st.checkbox("Evaluate Bubble size based on {}".format(select_bivariate_col_1))
                if y_col_bubble_size:
                    graph = bubble_chart(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1, size_based_on="y")
                    st.write(graph)
                else:
                    graph = bubble_chart(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
                    st.write(graph)
        elif select_bivariate_plot == "Step Plot":
            graph = step_plot(df=df, x_col=select_bivariate_col_2, y_col=select_bivariate_col_1)
            st.write(graph)


elif select_no_of_variables == "Multivariate Analysis - Analyze Multiple Columns":
    st.markdown("You have Selected : :rainbow[ {} ]".format(select_no_of_variables))
    st.write("Work on Multi-variate Analysis in this section")

    select_multivariate_plot = st.selectbox('Select the Type of Plot:', ['', "Scatter Matrix Plot"], 
                                   format_func=lambda x: 'Select an option' if x == '' else x)
    
    if select_multivariate_plot == "Scatter Matrix Plot":
        graph = scatter_matrix_plot(df=df)
        st.write(graph)


    

# bivariate
# relational analysis
#  ============================Data Visualization============================



#  ============================Encoding============================

#  ============================//Encoding============================


#  ============================Outliers============================

#  ============================//Outliers============================


#  ============================Skewness/Transformation============================

#  ============================//Skewness/Transformation============================

