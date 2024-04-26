import streamlit as st
import numpy as np
import pandas as pd
from annotated_text import annotated_text
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

tweets_file = "tweets.csv"
content_index_key = 'context_index'

st.markdown(
    """
    <style>
    .element-container:has(style){
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        background-color: lime;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if content_index_key not in st.session_state:
    st.session_state[content_index_key] = 0
    
def read_one_tweet():
    df = pd.read_csv(tweets_file)
    current_index = st.session_state[content_index_key]
    return df.iloc[current_index]['message']

def read_nz_slang():
    df = pd.read_csv("nz_slang.csv")
    df.set_index('phrase', inplace=True)
    return df

def Introduction():
    st.title("Introduction")
    st.write("This is the introduction page")
    st.write("this is something else <a href='#' id='my-link'>Click me</a>", unsafe_allow_html=True)
    st.button("my-link", on_click=click_button)

def click_button():
    st.write("Link clicked!")


def Moderation():
    message = read_one_tweet()
    st.title("Moderation")
    df_nz_slang = read_nz_slang()
    nz_slangs = df_nz_slang.index.tolist()
    found_slangs = []
    containter = st.container(height=500)
    
    col1, col2 = containter.columns([2, 1])
    with col1:
        st.subheader("Message")
        col3, col4, col5, col6 = col1.columns([1,1,1,1])
        col3.button("Prev", key="btn_prev", on_click=lambda: st.session_state.__setitem__(content_index_key, st.session_state[content_index_key] - 1))
        col4.button("Next", key="btn_next", on_click=lambda: st.session_state.__setitem__(content_index_key, st.session_state[content_index_key] + 1))
        col5.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        col5.button("Pass", key = "btn_pass")
        col6.button("Block", key="btn_block")
        st.write(message)

        for slang in nz_slangs:
            slang = slang.lower()
            if slang in message.lower():
                found_slangs.append(slang)
            
        
  
    # with col2:
    #     st.subheader("Information")
    #     st.write("Found NZ slangs:", found_slangs)
    #     text = preprocess(message)
    #     encoded_input = tokenizer(text, return_tensors='pt')
    #     output = model(**encoded_input)
    #     scores = output[0][0].detach().numpy()
    #     scores = softmax(scores)
    #     ranking = np.argsort(scores)
    #     ranking = ranking[::-1]
    #     st.write("Sentiment:", config.id2label[ranking[0]])
    #     for i in range(scores.shape[0]):
    #         l = config.id2label[ranking[i]]
    #         s = scores[ranking[i]]
    #         st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
        

        # ...

    with col2:
        st.subheader("Information")
        st.write("Found NZ slangs:", found_slangs)
        text = preprocess(message)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        st.write("Sentiment:", config.id2label[ranking[0]])
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            st.write(f"{i+1}) {l} {np.round(float(s), 4)}")
        
        # Bar chart
        labels = [config.id2label[ranking[i]] for i in range(scores.shape[0])]
        values = [float(scores[ranking[i]]) for i in range(scores.shape[0])]
        # Sort labels and values based on sentiment order: positive, neutral, negative
        sentiment_order = ['positive', 'neutral', 'negative']
        sorted_labels = [label for label in sentiment_order if label in labels]
        sorted_values = [values[labels.index(label)] for label in sorted_labels]
        plt.bar(sorted_labels, sorted_values)
        plt.xlabel('Sentiment')
        plt.ylabel('Probability')
        plt.title('Sentiment Analysis')
        st.pyplot(plt)
        # st.slider("Positive", min_value=0.1, max_value=1.0,step=0.0001, value=0.5, key="positive", on_change=lambda x: x*2)


def Content():
    st.title("Contents")
    df = pd.read_csv(tweets_file)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        height=500,
        use_container_width=True,
    )
    if st.button("Update"):
        edited_df.to_csv(tweets_file, index=False)


def Knowledge():
    st.title("Knowledge Base")
    df = read_nz_slang()
    st.write(df)


def main():
    with st.sidebar:
        st.header("Select the page")
    page = st.sidebar.radio(
        "", ["Introduction", "Moderation", "Contents", "Knowledge Base"]
    )
    if page == "Introduction":
        Introduction()
    elif page == "Moderation":
        Moderation()
    elif page == "Contents":
        Content()
    elif page == "Knowledge Base":
        Knowledge()


if __name__ == "__main__":
    main()
