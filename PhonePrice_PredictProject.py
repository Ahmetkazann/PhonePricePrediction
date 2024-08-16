#*******************************
## READ ME
# Open project integrated terminal then type terminal : streamlit run PhonePrice_PredictProject.py

#*******************************
import pandas as pd

def minmaxnormalize(dataset, column_name, new_min, new_max):
    column = dataset[column_name]
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    dataset[column_name] = normalized_column
    return dataset

df = pd.read_csv('PhoneDataset.csv')
df_copy = df.copy()

df_copy['RAM (MB)'] = df_copy['RAM (MB)'] / 1000
df_copy.rename(columns={"RAM (MB)": "RAM (GB)"}, inplace=True)

most_common_ratio = df_copy.apply(lambda x: x.value_counts().max() / len(x)) # The column is deleted if most of its values are the same


columns_to_keep = most_common_ratio[most_common_ratio < 0.95].index
columns_to_drop = most_common_ratio[most_common_ratio >= 0.95].index


df_copy = df_copy[columns_to_keep]


df_copy["4G/ LTE"] = df_copy["4G/ LTE"].map({"Yes": 2, "No": 1})

df_copy["3G"] = df_copy["3G"].map({"Yes": 2, "No": 1})

df_copy["GPS"] = df_copy["GPS"].map({"Yes": 2, "No": 1})


df_copy.drop('Name', axis = 1, inplace = True)
df_copy.drop('Brand', axis = 1, inplace = True)
df_copy.drop('Model', axis = 1, inplace = True)

dependent_variable = df_copy["Price"]
df_copy = df_copy.drop("Price", axis=1)
df_copy = df_copy.drop("Unnamed: 0", axis=1)

from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict

new_df_train, new_df_test, dependent_variable_train, dependent_variable_test = train_test_split(df_copy, dependent_variable, test_size = 0.20, random_state = 25)

from sklearn.linear_model import LinearRegression

mlrm = LinearRegression()

model = mlrm.fit(new_df_train, dependent_variable_train)

import numpy as np

predictions = model.predict(df_copy)
    
import streamlit as st

lorem = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type 
and scrambled it to make a type specimen book. It has survived not only five centuries"""
final_predict = 0

ICON = "logo.png"
st.sidebar.image(ICON, use_column_width=True)

st.title("Phone Price Prediction MLRM")

st.header("Choose values", divider=True)

Battery = st.sidebar.slider(
    "Battery Capacity (mAh)",
    float(df_copy['Battery capacity (mAh)'].min()),  # Minimum value
    float(df_copy['Battery capacity (mAh)'].max()),  # Maximum value
    float(df_copy['Battery capacity (mAh)'].mean())  # Default value
)

screen_size = st.sidebar.slider(
    "Screen size (inches)",
    float(df_copy['Screen size (inches)'].min()),
    float(df_copy['Screen size (inches)'].max()),
    float(df_copy['Screen size (inches)'].mean())
)

resx = st.sidebar.slider(
    "Resolution x",
    int(df_copy['Resolution x'].min()),
    int(df_copy['Resolution x'].max()),
    int(df_copy['Resolution x'].mean())
)

resy = st.sidebar.slider(
    "Resolution y",
    int(df_copy['Resolution y'].min()),
    int(df_copy['Resolution y'].max()),
    int(df_copy['Resolution y'].mean())
)

processor = st.sidebar.slider(
    "Processor Speed (GHz)",
    float(df_copy['Processor'].min()),
    float(df_copy['Processor'].max()),
    float(df_copy['Processor'].mean())
)

ram = st.sidebar.slider(
    "RAM (GB)",
    int(df_copy['RAM (GB)'].min()),
    int(df_copy['RAM (GB)'].max()),
    int(df_copy['RAM (GB)'].mean())
)

hdd = st.sidebar.slider(
    "Storage (GB)",
    int(df_copy['Internal storage (GB)'].min()),
    int(df_copy['Internal storage (GB)'].max()),
    int(df_copy['Internal storage (GB)'].mean())
)

akamera = st.sidebar.slider(
    "Rear Camera (MP)",
    int(df_copy['Rear camera'].min()),
    int(df_copy['Rear camera'].max()),
    int(df_copy['Rear camera'].mean())
)

okamera = st.sidebar.slider(
    "Front Camera (MP)",
    int(df_copy['Front camera'].min()),
    int(df_copy['Front camera'].max()),
    int(df_copy['Front camera'].mean())
)
gps = st.sidebar.selectbox("GPS", ["No", "Yes"])
sim = st.sidebar.selectbox("SIM Slots", ["1", "2", "4"])
g3 = st.sidebar.selectbox("3G", ["No", "Yes"])
lte = st.sidebar.selectbox("LTE", ["No", "Yes"])

gps = 0 if gps == 0 else 1
sim = 1 if sim == 1 else (2 if sim == 2 else 4)
g3 = 0 if g3 == 0 else 1
lte = 0 if lte == 0 else 1

final_phone = [int(Battery), float(screen_size), int(resx), int(resy), int(processor), int(ram), 
               int(hdd), int(akamera), int(okamera), int(gps), int(sim), int(g3), int(lte)]
columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 'Resolution y', 'Processor', 'RAM (GB)', 
           'Internal storage (GB)', 'Rear camera', 'Front camera', 'GPS', 'Number of SIMs', '3G', '4G/ LTE']

values = [[Battery, screen_size, resx, resy, processor, ram, 
          hdd, akamera, okamera, gps, sim, g3, lte]]

final_phone_df = pd.DataFrame(values, columns=columns)

final_predict = model.predict(final_phone_df)

final_predict = int(final_predict)

st.header("Predicted Price : " + str(final_predict), divider=True)

col1, col2, col3 = st.columns(3)
phone = df['Name'].sample(n=1).values[0]
with col1:
    st.header("This may interest you: ")
    st.markdown(f'<p style="color: {"red"};">{phone}</p>', unsafe_allow_html=True)

    if final_predict < 30000:
        st.image("2.jpg", caption="")
    else:
        st.image("1.jpg", caption="")

with col2:
    st.write(lorem)

with col3:
    st.write(lorem[:80])
    if st.button("More Details"):
        st.write("Here you can add more detailed information or additional actions related to the phone.")
        st.link_button("Dataset Link", "https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices")
st.title("DataFrame")

df['Predictions'] = predictions

df.set_index('Name', inplace=True)

df_copy2= df.head(40)
st.write("Dataframe")
st.write(df_copy2)


st.line_chart(df_copy2[['Price', 'Predictions']])

feature_names = new_df_train.columns
coefficients = model.coef_

coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient')

coefficients_df.set_index('Feature', inplace=True)

st.write("Model Coefficients (Line Chart)")
st.line_chart(coefficients_df['Coefficient'])



