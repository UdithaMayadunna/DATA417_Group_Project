import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import matplotlib.pyplot as plt
import re

# Maximum negative score threshold
MAX_NEGATIVE_SCORE = 0.01

# Define the highlighting function using HTML and CSS
def highlight_text(original_text, positions):
    highlighted_text = ""
    current_position = 0

    for start, end in positions:
        highlighted_text += original_text[current_position:start]
        highlighted_text += f'<span style="color: red;">{original_text[start:end]}</span>'
        current_position = end

    highlighted_text += original_text[current_position:]
    return highlighted_text

# function to match the text by dictionary
def matches_dictionary(text, df_dictionary):
    df_dictionary['Phrase'] = df_dictionary['Phrase'].str.lower()
    dictionary = df_dictionary.set_index("Phrase").to_dict()["Definition"]

    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in dictionary.keys()) + r')\b')
    matches = [(match.start(), match.end(), match.group(), dictionary[match.group()], df_dictionary[df_dictionary["Phrase"].str.lower() == match.group()]['Type'].iloc[0]) for match in pattern.finditer(text.lower())]
    return matches

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
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

tweets_file = "tweets.csv"
content_index_key = 'context_index'
negative_score_key = 'negative_score'

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

if negative_score_key not in st.session_state:
    st.session_state[negative_score_key] = 0.0

def read_one_tweet():
    df = pd.read_csv(tweets_file)
    current_index = st.session_state[content_index_key]
    return df.iloc[current_index]['message']

def read_dictionary():
    df = pd.read_csv("dictionary.csv")
    return df

def Introduction():
    st.title("Introduction")
    st.write("""
             Content moderation is critical to major global social media platforms such as Facebook
and Twitter. However, it faces challenges in effectively handling cultural differences. For
instance, content moderators may need help interpreting specific languages, such as
Maori or New Zealand slang, leading to inaccurate assessments. To address this issue, we
suggest a computer-aided approach that provides moderators with supplementary
information on these culturally specific terms.
             """)
    st.image("./public/diversity.png", caption="Handle Culture Diversity")

def Moderation():
    message = read_one_tweet()
    st.title("Moderation")
    df_dictionary = read_dictionary()
    matches = matches_dictionary(message, df_dictionary)
    positions = [(start, end) for start, end, _, _, _ in matches]
    highlighted_text = highlight_text(message, positions)
    
    container = st.container()
    
    col1, col2 = container.columns([2, 1])
    with col1:
        st.subheader("Message")
        col3, col4, col5, col6 = col1.columns([1, 1, 1, 1])
        col3.button("Prev", key="btn_prev", on_click=lambda: st.session_state.__setitem__(content_index_key, st.session_state[content_index_key] - 1))
        col4.button("Next", key="btn_next", on_click=lambda: st.session_state.__setitem__(content_index_key, st.session_state[content_index_key] + 1))
        col5.markdown('<span id="button-after"></span>', unsafe_allow_html=True) # use button-after css to change the color of button
        col5.button("Pass", key="btn_pass")
        col6.button("Block", key="btn_block")
        st.markdown(highlighted_text, unsafe_allow_html=True)
            
    with col2:
        st.subheader("Information")
        text = preprocess(message)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        st.write("Sentiment:", config.id2label[ranking[0]])
        for i in range(scores.shape[0]):
            label = config.id2label[ranking[i]]
            score = scores[ranking[i]]
            st.write(f"{i+1}) {label} {np.round(float(score), 4)}")
        
        # Update the cumulative negative score
        negative_score = scores[config.label2id['negative']]
        st.session_state[negative_score_key] += negative_score

        # Check if the cumulative negative score exceeds the threshold
        if st.session_state[negative_score_key] > MAX_NEGATIVE_SCORE:
            st.warning("You have seen a lot of negative content. Consider taking a break.")

        # Bar chart
        labels = [config.id2label[ranking[i]] for i in range(scores.shape[0])]
        values = [float(scores[ranking[i]]) for i in range(scores.shape[0])]
        sentiment_order = ['positive', 'neutral', 'negative']
        sorted_labels = [label for label in sentiment_order if label in labels]
        sorted_values = [values[labels.index(label)] for label in sorted_labels]
        plt.bar(sorted_labels, sorted_values)
        plt.xlabel('Sentiment')
        plt.ylabel('Probability')
        plt.title('Sentiment Analysis')
        st.pyplot(plt)
    st.write("Matched Dictionary Entries")
    st.dataframe(pd.DataFrame(matches, columns=['start', 'end', 'word', 'definition', 'type']))
    st.write("Cumulative Negative Score:", st.session_state[negative_score_key])
    st.write("Negative Score Threshold:", MAX_NEGATIVE_SCORE)  

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
    df = read_dictionary()
    st.write(df)

def main():
    with st.sidebar:
        st.header("Pages")
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
