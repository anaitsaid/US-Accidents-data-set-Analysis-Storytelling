from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import category_encoders as ce
import plotly.express as px
import pandas as pd
import numpy as np
# import io

# load dataset into "df" DataFrame
df = pd.read_csv('US_Accidents_Dec21_updated.csv')
print("--- Dataset Loaded ---")

'''# Save Info about raw data
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
with open("infoBefore.txt", "w", encoding="utf-8") as f:
    f.write(info)
df.describe().to_csv('descriptionBefore.csv')
# Missing data count + percentage
NumNaN = df.isnull().sum().sort_values(ascending=False)
PerNaN = df.isnull().sum().sort_values(ascending=False) / df.shape[0]
with open("missingDataBefore.csv", "w", encoding="utf-8") as f:
    f.write("Feature,Missing Data,Percentage\n")
    for idx in NumNaN.index.values:  # Checking for missing data
        line = idx + "," + str(NumNaN[idx]) + "," + "{:.2%}".format(PerNaN[idx])
        f.write(line + "\n")'''

# List for features to drop for dimensionality reduction
# Dropping ID column since it doesn't provide anything useful
# Dropping End_Time column as well due to irrelevance
# US Accidents data, so we are dropping the country column
# 61% of the Street Number is missing, so we are dropping it as well
# Dropping the Description columns because it's too vague -> len(df['Description'].unique()) == 1174563 (huge variation)
# Dropping location columns because ZipCode will be used to resume them
toDrop = ['ID', 'End_Time', 'Country', 'Number', 'Description', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
          'County', 'City', 'State', 'Street', 'Timezone', 'Airport_Code', 'Weather_Timestamp']

# Making sure all the Zipcodes follow the correct format (5 numbers - 4 numbers OR 5 numbers)
df['Zipcode'] = df['Zipcode'].str.extract(r'(^\d{5}(?:[-\s]\d{4})?$)', expand=False)
# We extract the three first digits of the Zipcode -> represent national area and sectional center
df['Zipcode'] = df['Zipcode'].map(lambda x: str(x[:3]), na_action="ignore")

# Filling these missing values of numerical data with the mean value not to damage the data when used
num_data = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
            'Wind_Speed(mph)', 'Precipitation(in)']  # List of numerical data columns
for elm in num_data:
    df[elm] = df[elm].fillna(df[elm].mean())

# List of boolean data columns
count_elm = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
             'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
             'Turning_Loop']
tr = []  # list for storing number of true values for each column
fl = []  # list for storing number of false values for each column
for elm in count_elm:
    count = df[elm].value_counts()
    indices = count.index.tolist()
    if len(indices) == 2:  # if the truth value varies between True & False
        tr.append(count[True])
        fl.append(count[False])
    else:  # if there's only one truth value we drop the elm as it doesn't offer valuable info
        toDrop.append(elm)
        count_elm.remove(elm)

# False to True count ratio in boolean type data
fig = go.Figure(data=[go.Bar(name='True', x=count_elm, y=tr),
                      go.Bar(name='False', x=count_elm, y=fl)])
fig.update_layout(barmode='group')
fig.write_html("fig1.html")

# Map of accident spread by severity
state_counts = df["State"].value_counts()
fig = go.Figure(data=go.Choropleth(locations=state_counts.index, z=state_counts.values.astype(float),
                                   locationmode="USA-states", colorscale="reds", colorbar=dict(title="Accidents")))
fig.update_layout(title_text="Number of US Accidents for each State", geo_scope="usa")
fig.write_html("fig2.html")

# Accidents count per state
state_counts = state_counts.to_frame()
state_counts.rename(columns={'State': 'Count'}, inplace=True)
state_counts.insert(0, "State", state_counts.index, True)
fig = px.bar(state_counts, x="State", y="Count", color="State", labels={"Count": "Accidents"}, text_auto=True,
             color_continuous_scale=px.colors.sequential.PuBuGn)
fig.write_html("fig3.html")

# Accidents count per severity
severity_counts = df["Severity"].value_counts()
severity_counts = severity_counts.to_frame()
severity_counts.rename(columns={'Severity': 'Count'}, inplace=True)
severity_counts.insert(0, "Severity", [2, 3, 4, 1], True)
severity_counts["Severity"] = severity_counts["Severity"].astype(str)
fig = px.bar(severity_counts, x='Severity', y='Count', labels={'Count': 'Accidents'}, color='Severity', text_auto=True)
fig.write_html("fig4.html")

# Accidents severity & count per state
severity_state = df[['State', 'Severity']].value_counts()
severity_state = severity_state.to_frame()
severity_state = severity_state.unstack(level='State')
severity_state.columns = severity_state.columns.droplevel()
severity_state = severity_state.transpose()
fig = px.bar(severity_state, labels={'value': 'Accidents'}, text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.write_html("fig5.html")

# Accidents count per state
street_counts = df["Street"].value_counts()
street_counts = street_counts.to_frame().head(20)
street_counts.rename(columns={'Street': 'Count'}, inplace=True)
street_counts.insert(0, "Street", street_counts.index, True)
fig = px.bar(street_counts, x="Street", y="Count", color="Street", labels={"Count": "Accidents"}, text_auto=True,
             color_continuous_scale=px.colors.sequential.PuBuGn)
fig.write_html("fig6.html")

# Accidents count per month
month = df['Start_Time'].str.split('-', expand=True)  # Get month value from Start_Time
month = month[1].value_counts()
month = month.to_frame()
month.rename(columns={1: 'Count'}, inplace=True)
month.insert(0, "Month", month.index, True)
fig = px.bar(month, x="Month", y="Count", color="Month", labels={"Count": "Accidents"}, text_auto=True,
             color_continuous_scale=px.colors.sequential.PuBuGn)
fig.update_layout(xaxis={'categoryorder': 'category ascending'})
fig.write_html("fig7.html")

# Accidents count per year
year = df['Start_Time'].str.split('-', expand=True)  # Get year value from Start_Time
year = year[0].value_counts()
year = year.to_frame()
year.rename(columns={0: 'Count'}, inplace=True)
year.insert(0, "Year", year.index, True)
fig = px.bar(year, x="Year", y="Count", color="Year", labels={"Count": "Accidents"}, text_auto=True,
             color_continuous_scale=px.colors.sequential.PuBuGn)
fig.update_layout(xaxis={'categoryorder': 'category ascending'})
fig.write_html("fig8.html")

# Accidents count per hour
hour = df['Start_Time'].str.split(' ', expand=True)
hour = hour[1].str.split(':', expand=True)
hour = hour[0].value_counts()
hour = hour.to_frame()
hour.rename(columns={0: 'Count'}, inplace=True)
hour.insert(0, "Hour", hour.index, True)
fig = px.bar(hour, x="Hour", y="Count", color="Hour", labels={"Count": "Accidents"}, text_auto=True,
             color_continuous_scale=px.colors.sequential.PuBuGn)
fig.update_layout(xaxis={'categoryorder': 'category ascending'})
fig.write_html("fig9.html")

# Accidents count per Weather variables (discrete data)
for elm in num_data:
    temprange = []
    templabel = []
    min = int(df[elm].min())
    max = int(df[elm].max())
    if max < 25:
        jump = int((abs(max) + abs(min)) / 24)
    elif max < 60:
        jump = int((abs(max) + abs(min)) / 50)
    else:
        jump = int((abs(max) + abs(min)) / 80)
    print(min, max)
    for i in range(min, max, jump):
        temprange.append(i)
        if i < (max - jump):
            label = str(i) + ' _ ' + str(i + jump)
            templabel.append(label)
    df['Range'] = pd.cut(x=df[elm], bins=temprange, labels=templabel)
    temp = df['Range'].value_counts()
    temp = temp.to_frame()
    temp.rename(columns={'Range': 'Count'}, inplace=True)
    temp.insert(0, "Range", temp.index, True)
    temp = temp.sort_values(by='Range')
    fig = px.bar(temp, x="Range", y="Count", color="Count", labels={"Count": "Accidents"}, title=elm,
                 color_continuous_scale=px.colors.sequential.Sunset)
    fig.write_html("fig_" + str(elm) + ".html")
    print("fig_" + str(elm) + ".html is ready")
del df['Range']

# Dropping boolean type data that has a low variance where: False values >>> True values
toDrop.extend(['Bump', 'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Traffic_Calming'])

# from string to datetime -> Get the year, month, day and hour separately from the Start_Time field and then drop it
df['Start_Time'] = pd.to_datetime(df["Start_Time"])
df['Year'], df['Month'], df['Day'], df['Hour'] = df['Start_Time'].dt.year, df['Start_Time'].dt.month,\
                                                 df['Start_Time'].dt.day, df['Start_Time'].dt.hour
toDrop.append('Start_Time')

# Feature Selection -> Dimensionality Reduction
df.drop(toDrop, inplace=True, axis=1)

# Data cleaning & Sampling
# We get the description of all object type data
obj_data = ['Wind_Direction', 'Weather_Condition', 'Side', 'Sunrise_Sunset',
            'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']

with open("categoricalDataDescriptionBefore.csv", "w", encoding="utf-8") as f:
    f.write('Feature,Count Of Unique Values')
    for elm in obj_data:
        f.write('\n' + elm + ',' + str(len(df[elm].unique())) + ',')

# "N" found in "Side" which is neither "R" right nor "L" left, so we change it to NaN
df.loc[df["Side"] == 'N', "Side"] = np.nan

# Grouping Wind_Direction data
Variation = ["CALM", "VAR", "East", "North", "South", "West"]
Equivalence = ["Calm", "Variable", "E", "N", "S", "W"]
for var, eq in zip(Variation, Equivalence):
    df.loc[df["Wind_Direction"] == var, "Wind_Direction"] = eq
df["Wind_Direction"] = df["Wind_Direction"].map(lambda x: x if len(x) != 3 else x[-2:], na_action="ignore")

# Fixing the Weather_Condition values to more structured values
Variation = ["Wintry", "Thunder|T-Storm|Vicinity|Tornado", "Snow|Sleet", "Rain|Drizzle", "Wind|Squalls", "Hail|Pellets",
             "Clear|Fair", "Cloud|Overcast", "Mist|Haze|Fog", "Sand|Dust", "Smoke|Volcanic Ash"]
Equivalence = ["Wintry", "Thunderstorm", "Snow", "Rain", "Windy", "Hail", "Clear", "Cloudy", "Fog", "Sand", "Smoke"]

df['Weather'] = [[] for _ in range(len(df.index))]
for var, eq in zip(Variation, Equivalence):
    weather = df.loc[df["Weather_Condition"].str.contains(var, na=False), "Weather"]
    # df.loc[df["Weather_Condition"].str.contains(var, na=False), "Weather"] += [eq]
    for val in weather:
        val.append(eq)
df["Weather"] = df["Weather"].str.join('/')
df.loc[df["Weather"] == '', "Weather"] = np.NaN

for weather in Equivalence:  # To create new columns for all different weather conditions
    df[weather] = 0
for var, eq in zip(Variation, Equivalence):
    count = df["Weather"].str.contains(var, na=False).sum()
    if count > 0:
        df.loc[df["Weather"].str.contains(var, na=False), eq] = 1

# Accidents Count & Severity per Weather Condition
weather_severity = df[['Severity', 'Weather']].value_counts()
weather_severity = weather_severity.to_frame()
weather_severity = weather_severity.unstack(level='Severity')
weather_severity.columns = weather_severity.columns.droplevel()
fig = px.bar(weather_severity, labels={'value': 'Accidents'}, text_auto=True)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.write_html("fig10.html")

# Missing data count % percentage after cleaning
NumNaN = df.isnull().sum().sort_values(ascending=False)
PerNaN = df.isnull().sum().sort_values(ascending=False) / df.shape[0]
with open("missingDataAfter.csv", "w", encoding="utf-8") as f:
    f.write("Feature,Missing Data,Percentage\n")
    for idx in NumNaN.index.values:  # Checking for missing data
        line = idx + "," + str(NumNaN[idx]) + "," + "{:.2%}".format(PerNaN[idx])
        f.write(line + "\n")


# The rest of the data is well categorized, and we already replaced NaN with mean for numerical values
# Now we drop rows with NaN values for none numerical values & Weather_Condition because we already replaced it with the manual encoding
df.dropna(inplace=True)
df.drop(["Weather_Condition", "Weather"], inplace=True, axis=1)
obj_data.remove('Weather_Condition')

# Due to previous processing -> duplicates may appear (some dropped or fixed fields may have been what differentiated rows)
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)

'''# Save Info about data after cleaning
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
with open("infoAfterSelection.txt", "w", encoding="utf-8") as f:
    f.write(info)
with open("categoricalDataDescriptionAfter.csv", "w", encoding="utf-8") as f:
    f.write('Feature,Count Of Unique Values')
    print('After Categorical data values')
    for elm in obj_data:
        f.write('\n' + elm + ',' + str(len(df[elm].unique())) + ',')
        print(elm, df[elm].unique())

df.describe().to_csv('descriptionAfter.csv')'''
df.to_csv("US_Accidents_AFTER.csv")

# Even after all this, our data size is still huge and inconsistent
# Data Sampling to reduce the size dealt with and balance data depending on target "Severity"
s = len(df[df["Severity"] == 1].index)  # Size of the smallest Severity occurrence
dfSample = df[df["Severity"] == 1]
for sev in range(2, 5):  # Conditional Sampling + Constant Rate Sampling
    dfSample = pd.concat([dfSample, df[df["Severity"] == sev].sample(s * 2, random_state=42)[::2]], ignore_index=True)
dfSample.to_csv('Sample.csv')

# Normalizing numerical data
dfNorm = dfSample.copy()
num_data.extend(['Year', 'Month', 'Day', 'Hour'])
# define min max scaler
scaler = MinMaxScaler()
toScale = dfSample[num_data]
# transform data
scaled = pd.DataFrame(scaler.fit_transform(toScale), columns=toScale.columns)
# update df
dfNorm.update(scaled)

# Finally, we numerize the remaining categorical data
dfNorm = pd.get_dummies(dfNorm, columns=['Side', 'Wind_Direction'], drop_first=True)
dfNorm = pd.get_dummies(dfNorm, columns=['Sunrise_Sunset', 'Civil_Twilight',
                                         'Nautical_Twilight', 'Astronomical_Twilight'],
                        prefix=['Sun', 'Civ', 'Nau', 'Ast'], drop_first=True)
# True & False replaced by 1 & 0
dfNorm = dfNorm.replace([True, False], [1, 0])
# Zipcode binary encoded
binary_encoder = ce.binary.BinaryEncoder()
binZip = binary_encoder.fit_transform(dfNorm['Zipcode'])
dfNorm = pd.concat([dfNorm, binZip], axis=1).drop('Zipcode', axis=1)
dfNorm.to_csv('SampleNormalized&Encoded.csv')
