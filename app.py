import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import collections
import string

@st.cache_data
def load_data():
    df = pd.read_csv('FLORES200.tok.csv')
    df['ID'] = range(1, len(df) + 1)
    return df

def reload_example_text_data():
    frames = []  # List to store dataframes for each language

    for lang in languages:
        # Filter the data for the selected language
        lang_data = data[data['language'] == lang]

        # Pick a random row from the filtered data
        random_row = lang_data.sample(1)

        # Append to the frames list
        frames.append(random_row)

    # Concatenate all the dataframes in the frames list
    tempdf = pd.concat(frames, ignore_index=True)
    
    tempdf.rename(columns={'language': 'Language'}, inplace=True)
    tempdf.set_index('Language', inplace=True)
    tempdf = tempdf[['sentence', "token_length"]]
    tempdf.columns = ['Sentence','Num Tokens']
    tempdf.sort_values(by='Num Tokens', inplace=True)
    st.session_state.examplesdf = tempdf

def get_common_tokens_for_languages(langs, n=10):
    tokens = []
    exclude_tokens = ['eng_Latn', '</s>'] + list(string.punctuation)
    
    for lang in langs:
        lang_data = data[data['language'] == lang]
        lang_tokens = ' '.join(lang_data['tokens']).split()
        
        # Filter out excluded tokens
        filtered_tokens = [token for token in lang_tokens if token not in exclude_tokens]
        
        token_freq = collections.Counter(filtered_tokens)
        common_tokens_for_lang = token_freq.most_common(n)
        tokens.append((lang, common_tokens_for_lang))
    return tokens

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

st.set_page_config(layout="wide")

tokenizer_name = "facebook/nllb-200-distilled-600M"

with st.sidebar:
    st.subheader('Tokenizer')
    link = f"Tokenized using [{tokenizer_name}]"
    st.markdown(link)
        
    st.subheader('Data')
    with st.spinner('Loading dataset...'):
        data = load_data()
        st.success(f'Data loaded: {len(data)} records. 997 sentences for {len(data["language"].unique())} languages.')

    with st.expander('Data Source'):
        st.write("The data in this dashboard corresponds to the tokenized dev set of the FLORES 200 dataset.")

    st.subheader('Languages')
    languages = st.multiselect(
        'Select languages',
        options=sorted(data['language'].unique()),
        default=["eng_Latn",  "cat_Latn",  "hrv_Latn",  "slv_Latn"],
        max_selections=10
    )

    st.subheader('Figure')
    show_hist = st.checkbox('Show histogram', value=False)
    distplot_mode = st.radio("Distplot Mode", ["density", "frequency"])

    st.subheader('Common tokens')
    n_tokens = st.slider('Number of tokens to show per language', 5, 50, 10)
        
with st.container():
    st.subheader(f'Tokenizer `{tokenizer_name}`')

grid = make_grid(1,2)

with grid[0][0]:
    st.subheader('Stats')
    # Calculate and display the stats for each language
    for _lang in languages:
        data_lang = data[data['language'] == _lang]
        mean = np.mean(data_lang['token_length'])
        median = np.median(data_lang['token_length'])
        std = np.std(data_lang['token_length'])
        min_val = np.min(data_lang['token_length'])
        max_val = np.max(data_lang['token_length'])
        stats_str = f"**{_lang}** - {mean:.2f} ± {std:.2f} (median: {median} min: {min_val}, max: {max_val})"
        st.write(stats_str)

with grid[0][1]:
    st.subheader('Token Distribution')
    # Token distribubtion visualization
    if distplot_mode == "density":
        histnorm_mode = "probability density"
    else:  # "frequency"
        histnorm_mode = None  # default is frequency when histnorm is None

    subset_data = [data[data['language'] == lang]['token_length'].tolist() for lang in languages]
    fig = ff.create_distplot(subset_data, group_labels=languages, show_hist=show_hist, histnorm=histnorm_mode)
        
    fig.update_layout(
        xaxis_title="Number of Tokens",
        yaxis_title="Density" if histnorm_mode == "probability density" else "Frequency",
        height=500
        )
    st.plotly_chart(fig, use_container_width=True)

# Example sentences
st.subheader('Example Texts')
reload_example_text_data()
if st.button("🔄 Randomly sample"):
    reload_example_text_data()
st.dataframe(st.session_state.examplesdf,use_container_width=True)

st.subheader('Common Tokens')

# Fetch common tokens
common_tokens_by_language = get_common_tokens_for_languages(languages, n_tokens)

# Determine the number of columns based on selected languages
n_languages = len(common_tokens_by_language)

if n_languages == 1:
    n_cols = 1
elif n_languages <= 4:
    n_cols = 2
else:
    n_cols = 4

# Calculate the number of rows we need
n_rows = -(-n_languages // n_cols)  # This is a ceil division

for i in range(n_rows):
    cols = st.columns(n_cols)

    for j in range(n_cols):
        idx = n_cols * i + j
        if idx < n_languages:
            lang, common_tokens = common_tokens_by_language[idx]
            tokens, freqs = zip(*common_tokens)
            fig = px.bar(x=freqs, y=tokens, orientation='h', labels={'x': 'Frequency', 'y': 'Token'}, title=f'Most Common Tokens for {lang}')
            cols[j].plotly_chart(fig)


with st.expander("About the project"):
    st.write("The purpose of this project is to compare the tokenization length for different languages on FLORES200. This is part of a larger effor to OS checkpoints for the 202 languages of the NLLB project.")