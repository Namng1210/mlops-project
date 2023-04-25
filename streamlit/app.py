from pathlib import Path

import pandas as pd

import streamlit as st
from config import config
from tagifai import main, utils

# title
st.title("Mlops course Made with ML")


@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
    df = pd.read_csv(projects_fp)
    return df


# Section
st.header("🔢 Data")
df = load_data()
st.text(f"Projects (count: {len(df)})")
st.write(df)


st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag_class = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag_class])
tag_slice = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag_slice])

st.header("🚀 Inference")
text = st.text_input("Enter text:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Enter run id :", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
