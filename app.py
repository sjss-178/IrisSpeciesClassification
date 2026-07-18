# import streamlit as st
# import pandas as pd
# import numpy as np

# #Title of the application
# st.title("Hello Streamlit")

# # write some content
# st.write("This is a streamlit application")

# df=pd.DataFrame([[1,2,3],[4,5,6]],columns=["A","B","C"])

# #printing a dataframe using streamlit
# st.write(df)

# nums=np.random.randint(0,10,(5,5))

# #line chart using streamlit

# st.line_chart(nums,x_label="x-axis-data",y_label="y-axis-data")

# #taking input using streamlit

# name = st.text_input("Enter Your Name")

# if name:
#     st.write(name)

# #slider using stream lit
# age = st.slider("Select your age : ",0,100,25)



# if age:
#     st.write(f"you are {age} years old")


# uploaded = st.file_uploader("choose a CSV file")

# if uploaded is not None:
#     data=pd.read_csv(uploaded)
#     st.write(data)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


st.title("Random Forest Classifier For Iris Species DataSet")

@st.cache
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df["species"]=iris.target
    return df, iris.target_names

df,target_names=load_data()


model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df["species"])

st.sidebar.title("Input Features")
sl=st.sidebar.slider("sepal length (cm)",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sw=st.sidebar.slider("sepal width (cm)",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
pl=st.sidebar.slider("petal length (cm)",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
pw=st.sidebar.slider("petal width (cm)",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

prediction = model.predict([[sl,sw,pl,pw]])
predicted_species= target_names[prediction[0]]
st.write("Input tuple : ")
st.write(pd.DataFrame([[sl,sw,pl,pw]],columns=df.columns[:-1]))
st.title(f"Predicted Species  = {predicted_species}")


