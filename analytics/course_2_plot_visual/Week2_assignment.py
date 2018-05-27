#from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')

df = pd.read_csv('/Users/thayhuynh/Desktop/Projects/neural-networks-and-deep-learning/analytics/course_2_plot_visual/temperature.csv')

df = df[df['Date'] != '2008-02-29']
df = df[df['Date'] != '2012-02-29']

df['Date']=pd.to_datetime(df['Date'])
#df['Date']=df['Date'].apply(lambda x: x.replace(year=2015))
df['Date'] = df['Date'].apply(lambda d: d.strftime('%m-%d'))
grouped_df = df[df['Date'] != '02-29'].groupby('Date').agg({'Data_Value': [max,min]})
#grouped_df.head
#grouped_df = df[df['Date'] == '2005-11-11'].groupby(df['Date'].map(lambda d: datetime.strptime(d, '%Y-%m-%d')))
maxes = grouped_df['Data_Value']['max']
mines = grouped_df['Data_Value']['min']

dates = df['Date'].unique()
#grouped_df
m = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov','Dec']
# This is the vital step. It will create a list of day numbers corresponding to middle of each month i.e. 15(Jan), 46(Feb), ...
ticks = [(dt.date(2017,m,1)-dt.date(2016,12,15)).days for m in range(1,13)]

plt.figure()
#plt.interactive(False)
#print dates.sort_values()
#ax = plt.gca()
#ax.set_xticks(ticks)
#ax.set_xticklabels(m)

linear_data = np.array([1,2,3,4,5,6,7,8])
#plt.plot(linear_data, '-o')


plt.plot(maxes.values, '-o')

#ax = plt.gcf()
#ax.set_xticks(ticks)
#ax.set_xticklabels(m)
#plt.xlabel(m, ticks)

plt.show()

#plt.plot(dates, maxes, dates, mines, '-o')
#grouped_df.describe
#for key, item in grouped_df:
 #   print grouped_df.get_group(key)