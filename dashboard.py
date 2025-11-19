import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Titanic Data Analysis Dashboard")

df = pd.read_csv('titanic.csv')

st.write("### Dataset Overview")
st.write(df.head())

st.write("### Survival by Sex")
sns.countplot(data=df, x='Sex', hue='Survived')
st.pyplot(plt.gcf())

st.write("### Age Distribution")
df['Age'].hist()
st.pyplot(plt.gcf())

st.write("### Survival by Class")
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
st.pyplot(plt.gcf())

st.write("### Age vs Fare")
sns.scatterplot(data=df, x='Age', y='Fare')
st.pyplot(plt.gcf())

st.write("### Correlation Matrix")
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
st.pyplot(plt.gcf())

st.write("### Pair Plot")
sns.pairplot(df, hue='Survived')
st.pyplot(plt.gcf())

st.write("### Age by Class")
sns.boxplot(data=df, x='Pclass', y='Age')
st.pyplot(plt.gcf())

st.write("### Age by Sex")
sns.violinplot(data=df, x='Sex', y='Age')
st.pyplot(plt.gcf())
