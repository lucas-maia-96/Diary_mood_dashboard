import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import plotly.express as px

# create a list of all the files txt in the directory
files = glob.glob("diary/*.txt")
# sort a list of strings
files.sort()

analyser = SentimentIntensityAnalyzer()
negs = []
pos = []
dates = []

for dia in files:
    with open(dia, 'r') as f:
        text = f.read()

    score = analyser.polarity_scores(text)
    pos.append(score['pos'])
    negs.append(score['neg'])

    dates.append(dia.strip('diary/').strip('.txt'))

st.title("Diary Tone")

st.subheader("Positivity")

pos_figure = px.line(x=dates, y=pos, labels={"x": "Date", "y": "Positivity"})

st.plotly_chart(pos_figure)

st.subheader("Negativity")

pos_figure = px.line(x=dates, y=negs, labels={"x": "Date", "y": "Negativity"})

st.plotly_chart(pos_figure)
