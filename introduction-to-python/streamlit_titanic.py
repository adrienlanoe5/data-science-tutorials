import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/adrie/PycharmProjects/data-science-tutorials/data/train.csv")
st.markdown("# dataset titanic")
st.dataframe(df.head())
st.markdown("## valeurs manquantes")
st.dataframe(df.isnull().sum())
st.markdown("## Nombre de survivants")
st.dataframe(df.groupby('Survived')['PassengerId'].agg(['count']))
st.bar_chart(df.groupby('Survived')['PassengerId'].agg(
    ['count']
))
st.markdown("## Survival rate depending on genre")
st.dataframe(df.groupby(
    ['Survived', 'Sex']
)['PassengerId'].agg(['count']))
st.dataframe(df.groupby(
    ['Survived', 'Sex']
)['PassengerId'].agg(['count']).unstack())
fig, ax = plt.subplots(figsize=(10, 10))

df.groupby(["Survived", "Sex"])["PassengerId"].count().unstack().plot(
    kind="bar", ax=ax, figsize=(10, 10))
st.pyplot(fig)

st.markdown("## Survival rate depending on Ticket class")
fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(["Survived", "Pclass"])["PassengerId"].count().unstack().plot(
    kind="bar", ax=ax, figsize=(10, 10))
st.pyplot(fig)

st.markdown("## Survival rate depending on age")

df['generation'] = pd.cut(df['Age'], 8)
fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(
    ['Survived', 'generation']
)['PassengerId'].count().unstack().plot(kind ='bar',ax=ax, figsize = (10, 10))
st.pyplot(fig)

st.markdown("## Survival rate depending on Fare")

df['fare_category'] = pd.cut(df['Fare'], 12)

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(
    ['Survived', 'fare_category']
)['PassengerId'].count().unstack().plot(kind ='bar', ax=ax, figsize = (10, 10))
st.pyplot(fig)

st.markdown("## What about correlations ?")

st.dataframe(df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].corr())

st.markdown("## Best size of the boat")


def cote_bateau(arg):
    if pd.isna(arg) :
        return pd.NA
    if not arg[-1].isdigit():
        return pd.NA
    elif int(arg[-1])//2==0:
        return "starboard"
    else :
        return "port"

df["size_boat"]=df["Cabin"].apply(cote_bateau)

fig, ax = plt.subplots(figsize=(10, 10))
df.groupby(
    ['Survived', 'size_boat']
)['PassengerId'].count().unstack().plot(kind ='bar', ax=ax, figsize = (10, 10))
st.pyplot(fig)

def deck_number(arg):
    if pd.isna(arg) :
        return pd.NA
    else :
        return arg[0]