from enum import unique
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("NBA Player Stats Explorer")

st.markdown("""
The Purpose of This Application Is To Perform Simple Web Scraping Of NBA Data & Statistics.
* **Python Libraries:** base64, pandas, streamlit
* **Data Source:** [basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header("USER INPUT FEATURES")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950,2022))))

# web scraping of nba player stats
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]

    # deletes repeating headers
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)

    return playerstats

playerstats = load_data(selected_year)

# team selection sidebar
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team,sorted_unique_team)

# position selection sidebar
# pg(point guard) | sg(shooting guard) | sf(small forward) | pf(power forward) | c(center)
unique_pos = ['PG','SG','SF','PF','C']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# data filtering
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Show Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + 'columns.')
st.dataframe(df_selected_team)

# download csv file for nba player stats
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'

    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# heatmap
if st.button('INTERCORRELATION HEATMAP'):
    st.header('INTERCORRELATION MATRIX HEATMAP')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)

    st.pyplot(plt)