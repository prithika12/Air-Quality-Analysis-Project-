#Import the required modules and load the time-series dataset on air quality
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

months = ['Jan','Feb','March','April','May','June','July','Aug','Sept','Oct','Nov','Dec']

# Loading the dataset.
csv_file = 'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/air-quality/AirQualityUCI.csv'
df = pd.read_csv(csv_file, sep=';')

# Dropping the 'Unnamed: 15' & 'Unnamed: 16' columns.
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], axis=1) 

# Dropping the null values.
df = df.dropna()

# Creating a Pandas series containing 'datetime' objects.
dt_series = pd.Series(data = [item.split("/")[2] + "-" + item.split("/")[1] + "-" + item.split("/")[0] for item in df['Date']], index=df.index) + ' ' + pd.Series(data=[str(item).replace(".", ":") for item in df['Time']], index=df.index)
dt_series = pd.to_datetime(dt_series)

# Remove the Date & Time columns from the DataFrame and insert the 'dt_series' in it.
df = df.drop(columns=['Date', 'Time'], axis=1)
df.insert(loc=0, column='DateTime', value=dt_series)

# Get the Pandas series containing the year values as integers.
year_series = dt_series.dt.year

# Get the Pandas series containing the month values as integers.
month_series = dt_series.dt.month

# Get the Pandas series containing the day values as integers.
day_series = dt_series.dt.day

# Get the Pandas series containing the days of a week, i.e., Monday, Tuesday, Wednesday etc.
day_name_series = dt_series.dt.day_name()

# Add the 'Year', 'Month', 'Day' and 'Day Name' columns to the DataFrame.
df['Year'] = year_series
df['Month'] = month_series
df['Day'] = day_series
df['Day Name'] = day_name_series

# Sort the DataFrame by the 'DateTime' values in the ascending order. Also, display the first 10 rows of the DataFrame.
df = df.sort_values(by='DateTime')

# Create a function to replace the commas with periods in a Pandas series.
def comma_to_period(series):
    new_series = pd.Series(data=[float(str(item).replace(',', '.')) for item in series], index=df.index)
    return new_series

# Apply the 'comma_to_period()' function on the ''CO(GT)', 'C6H6(GT)', 'T', 'RH' and 'AH' columns.
cols_to_correct = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH'] # Create a list of column names.
for col in cols_to_correct: # Iterate through each column
    df[col] = comma_to_period(df[col]) # Replace the original column with the new series.

# Remove all the columns from the 'df' DataFrame containing more than 10% garbage value.
df = df.drop(columns=['NMHC(GT)', 'CO(GT)', 'NOx(GT)', 'NO2(GT)'], axis=1)

# Create a new DataFrame containing records for the years 2004 and 2005.
aq_2004_df = df[df['Year'] == 2004]
aq_2005_df = df[df['Year'] == 2005]

# Replace the -200 value with the median values for each column having indices between 1 and -4 (excluding -4) for the 2004 year DataFrame.
for col in aq_2004_df.columns[1:-4]:
  median = aq_2004_df.loc[aq_2004_df[col] != -200, col].median()
  aq_2004_df[col] = aq_2004_df[col].replace(to_replace=-200, value=median)

# Repeat the same exercise for the 2005 year DataFrame.
for col in aq_2005_df.columns[1:-4]:
  median = aq_2005_df.loc[aq_2005_df[col] != -200, col].median()
  aq_2005_df[col] = aq_2005_df[col].replace(to_replace=-200, value=median)

# Group the DataFrames about the 'Month' column.
group_2004_month = aq_2004_df.groupby(by='Month')
group_2005_month = aq_2005_df.groupby(by='Month')

# Concatenate the two DataFrames for 2004 and 2005 to obtain one DataFrame.
df = pd.concat([aq_2004_df, aq_2005_df])

# Information of the DataFrame.
df.info()


def group_by_months():
  """Groups the data based on the years and months"""
  #Group the records for the 2004 DataFrame together by month
  group_2004_month=aq_2004_df.groupby(by='Month')
  print('Grouping the records for the 2004 DataFrame together by month',group_2004_month)

  #Group the records for the 2005 DataFrame together by month.
  group_2005_month = aq_2005_df.groupby(by='Month')
  print('Records for the 2005 DataFrame together by month',group_2005_month)

  #Get the descriptive statistics for March 2004.
  print('Descriptive statistics for March 2004',group_2004_month.get_group(3).describe())

  #Get the descriptive statistics for March 2005.
  print('Descriptive statistics for March 2005',group_2005_month.get_group(3).describe())

  #Get mean, standard deviation and median for all the months for the year 2004.
  print('Mean, standard deviation and median for all the months for the year 2004',group_2004_month.agg(func=('mean','median','std')))

  #Get mean, standard deviation and median for all the months for the year 2005.
  print('Mean, standard deviation and median for all the months for the year 2005',group_2005_month.agg(func=['mean',"median","std"]))


def winter_season_data():
  #Get mean, standard deviation and median values for the winter season of 2004.
  print('Mean, standard deviation and median values for the winter season of 2004',group_2004_month.agg(func=['mean','std','median']).loc[[3,11,12],:])

  # Get mean, standard deviation and median values for the winter season of 2004 without the 'Year' & 'Day' columns.
  print('Mean, standard deviation and median values for the winter season of 2004 without the Year & Day columns',group_2005_month.agg(func=['mean','std','median']).loc[[1,2,3],:])


def matplotlib_CO_2004():
  """matplotlibplot for CO concentration in 2004"""

  #to specify the background of the graph 
  plt.style.use('ggplot')
  # to specify the width& height of the graph
  plt.figure(figsize=(15,5))
  # assign a title to the graph
  plt.title('Monthly Median CO Concentration In 2004')
  # create the line plot for the monthly median concentration for carbon monoxide. 'g-o' gives the green color to the plot 
  plt.plot(group_2004_month.median().index,group_2004_month.median()['PT08.S1(CO)'],'g-o')
  # label for the x-axis
  plt.xlabel('Month')
  # label for the y-axis
  plt.ylabel('Sensor Response')
  plt.grid(True)
  plt.show()

def matplotlib1_CO_2004():
  """Creates a DataFrame for 2004 containing median values and grouped by months that are common to 2004 and 2005 observations"""
  # data available in the air quality df for the year 2004 
  month_2004 = ['March','April','May','June','July','August','Sept','Oct','Nov','Dec']
  # data available in the air quality df for the year 2005
  month_2005 = ['Jan','Feb','March','April']
  # gives the dark background image for the graph 
  plt.style.use('seaborn-dark')
  plt.figure(figsize=(15,5))
  plt.title('DataFrame for 2004 containing median values')
  plt.plot(group_2004_month.median().index,group_2004_month.median()['PT08.S1(CO)'],'ro--',label=2004)
  # xticks is used to replace the numbers on the x-axis with the month names 
  plt.xticks(ticks=group_2004_month.median().index,labels=month_2004)
  plt.xlabel('Months')
  plt.ylabel('Sensor Response')
  plt.show()

def comparism_CO_concentration_2004_2005():
  """Creates a line plot for the monthly median CO concentrations for both the years. Use the 'seaborn-dark' style this time""" 
  month_2004 = ['March','April','May','June','July','August','Sept','Oct','Nov','Dec']
  month_2005 = ['Jan','Feb','March','April']
  # set the bg,title,size and labels like done in the above plots
  plt.style.use('seaborn-dark')
  plt.figure(figsize=(15,5))
  plt.title('DataFrame for 2005 containing median values')
  plt.plot(group_2004_month.median().index,group_2004_month.median()['PT08.S1(CO)'],'ro--',label=2004)
  plt.plot(group_2005_month.median().index,group_2005_month.median()['PT08.S1(CO)'],'ro--',label=2005)
  plt.xticks(ticks=group_2005_month.median().index,labels=month_2005)
  plt.xlabel('Months')
  plt.ylabel('Sensor Response')
  plt.show()


def line_plot(style,width,height,x_series,y_series,year,color):
  """Create a user-defined function to make a line plot between two series & also allows a user to change the plot attributes on fly"""
  plt.style.use(style)
  plt.figure(figsize=(width,height))
  plt.title(f"\n Time Series plot for {y_series.name} in {year}")
  plt.plot(x_series,y_series,color)
  plt.grid()
  plt.show()


def boxplot():
  """Creates a boxplot for month-wise variation in temperature split by year""" 
  plt.figure(figsize=(15,5))
  # boxplot is available in the seaborn module which shows the trend of data with months on the x-axis and temp on y-axis. 
  # hue= 'Year' seperates the year 2004 & 2005 dataframe 
  sns.boxplot(x='Month',y='T',hue='Year',data=df)
  plt.show()

def customized_barplots():
  #to loop through the months in 2004 based on the df 
  for i in group_2004_month.median().iloc[:,:-2]:
    plt.figure(figsize=(15,5))
    plt.title(f"monthly_median {i} variation in 2004")
    # creates the bar graph with months from march to dec on x-axis & temp on y-axis with width of each slice of bar plot as 0.6 
    plt.bar(x=np.arange(3,13),height=group_2004_month.median()['T'],width=0.6)
    plt.xticks(ticks=np.arange(3,13),labels=months)
    plt.show()

#Calculate the R value between all the air pollutants, temperature, relative & absolute humidity columns in the 'df' DataFrame.
corr_df = df.iloc[:,1:-4].corr()
corr_df

#super class for the air quality 
class AirQuality:
  def __init__(self,width,height,style):
    self.width = width 
    self.height = height 
    self.style = style
    
  def comparison_2004_2005_barplots(self):
    """ creates bivariate bar plots to visualise and compare the monthly median temperature variation for the year 2004 and 2005"""
    plt.figure(figsize=(self.width,self.height))
    plt.bar(x=np.arange(3,13),height=group_2004_month.median()['T'],width=0.6)
    plt.bar(x=np.arange(1,5),height=group_2005_month.median()['T'],width=0.6)
    plt.style.use(self.style)
    plt.title('Monthly median temperature variation in 2004 and 2005')
    months = ['Jan','Feb','March','April','May','June','July','Aug','Sept','Oct','Nov','Dec']
    plt.xticks(ticks=np.arange(1,13),labels=months)
    plt.xlabel('Months')
    plt.ylabel('T')
    plt.show()

# AirQuality_heat_map extends the base class AirQuality 
class AirQuality_heat_map(AirQuality):
  def __init__(self,width,height,style,dpi,cmap):
    super().__init__(width,height,style)
    self.dpi = dpi
    self.cmap = cmap
 
  def heat_map(self):
    #Change the colour scheme of the cells of the above heatmap to 'yellow-green-blue' by passing the 'YlGnBu' value to the 'cmap' parameter.
    # dpi- dots per inch; used to show the image quality in pixels 
    plt.figure(figsize=(self.width,self.height),dpi=self.dpi)
    # it is used to show the relation between each and every pollutant in the air quality analysis project df 
    sns.heatmap(data=corr_df,annot=True,cmap=self.cmap)
    plt.show()

aq = AirQuality(15,5,'seaborn-dark')
aq.comparison_2004_2005_barplots()
aq1 = AirQuality_heat_map(15,5,'seaborn-dark',108,'YlGnBu')
aq1.heat_map()


def regression_plot_CO_O3():
  """Creates a regression plot for the carbon monoxide and ozone columns"""
  plt.figure(figsize=(15,5))
  # regplot-regression plot; separates the relationship between the 2 pollutants 
  sns.regplot(x='PT08.S1(CO)',y='PT08.S5(O3)',data=df,color='brown')
  plt.show()


def regression_plot_CO_NOx():
  """Creates a regression plot for the carbon monoxide and nitrogen oxide columns"""
  plt.figure(figsize=(15,5))
  sns.regplot(x='PT08.S1(CO)',y='PT08.S3(NOx)',data=df,color='blue')
  plt.show() 

def pair_plot():
  """Creates the scatter plots for the numeric columns of the 'df' DataFrame in one go"""
  # pairplot shows the graphical relationship between each and every pollutant 
  sns.pairplot(df.iloc[:,1:-4])
  plt.show()


def pie_chart():
  #Get the month names from the 'DateTime' column for each record.
  df['DateTime'].dt.month_name()
  #Add the 'Month Name' column to the 'df' DataFrame and print the first five rows of the updated DataFrame.
  df['Month Name'] = df['DateTime'].dt.month_name()
  year_slices = df['Year'].value_counts()*100/df.shape[0]
  year_labels = ['Year 2004','Year 2005']
  #Add 3D effect to the pie.
  explode = [0,0.15]
  plt.figure(dpi=108)
  plt.title('The percentage of data collected in 2004 and 2005')
  # creates the pie chart with year 2004 and 2005 as labels with red color edges and assigns the percentage of data according to data available
  plt.pie(year_slices,labels=year_labels,explode=explode,wedgeprops={'edgecolor':'red'},autopct='%1.1f%%',shadow=True)
  plt.show()
  #Create a pie chart for the 2005 displaying all the months. Label the slices with the month names.
  data = df.loc[df['Year'] == 2005, 'Month Name'].value_counts()
  explode = np.linspace(0, 0.5, 4) # Shift the slices away from the centre of the pie 
  # specifies the image quality in pixels 
  plt.figure(dpi=108)
  plt.title("Percentage of Data Collected in 2005")
  plt.pie(data, labels=data.index, 
          explode=explode, autopct='%1.2f%%',
          startangle=30, # The first slice will be placed at an angle of 30 degrees w.r.t. to the horizontal axis in the anti-clockwise direction.
          shadow=True,
          wedgeprops={'edgecolor':'r'})

  plt.show()


group_by_months()
winter_season_data()
matplotlib_CO_2004()
matplotlib1_CO_2004()
comparism_CO_concentration_2004_2005()
boxplot()


#Create a line plot for all the air pollutants concentrations, temperature, relative & absolute humidity for the year 2004.
# This time use the 'seaborn-dark' style and red colour.
for i in aq_2004_df[1:-4]:
  line_plot('seaborn-dark',15,5,aq_2004_df['DateTime'],aq_2004_df[i],2004,'r')

#Create a line plot for all the air pollutants concentrations, temperature, relative and absolute humidity for the year 2005.
# This time use the 'dark_background' style and yellow colour.
for i in aq_2005_df[1:-4]:
  line_plot('seaborn-dark',15,5,aq_2005_df['DateTime'],aq_2005_df[i],2005,'b')

customized_barplots()
regression_plot_CO_O3()
regression_plot_CO_NOx()
pie_chart()
pair_plot()
