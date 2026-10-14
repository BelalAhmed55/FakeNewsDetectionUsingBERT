"""
Fake News Headline Detector - Streamlit inference app.

Inference only: the BERT model is never trained here. The trained weights
(c2_new_model_weights.pt) are downloaded from the Hugging Face Hub and loaded
into the same BERT_Arch used in the notebook.
"""
import os

import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from transformers import AutoModel, BertTokenizerFast

# ---------------------------------------------------------------------------
# Settings (all taken from the training notebook)
# ---------------------------------------------------------------------------
HF_REPO_ID = "belalahmed11/bert-text-classifier-weights"
WEIGHTS_FILENAME = "c2_new_model_weights.pt"
PRETRAINED_NAME = "bert-base-uncased"   # same pretrained model + tokenizer as the notebook
MAX_LENGTH = 15                         # MAX_LENGHT in the notebook: titles are cut to 15 tokens
CLASS_NAMES = ["Real", "Fake"]          # notebook label_map = {"Fake": 1, "True": 0} -> index 0 = Real, 1 = Fake
MIN_WORDS = 3                           # same minimum as the notebook's Gradio app
MAX_CHARS = 1000


# ---------------------------------------------------------------------------
# Model architecture - copied unchanged from the notebook
# ---------------------------------------------------------------------------
class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x


# ---------------------------------------------------------------------------
# Loading (runs once per server process, then cached)
# ---------------------------------------------------------------------------
def get_hf_token():
    """Token is only needed if the Hugging Face repo is private. Never hardcoded."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:       # no secrets configured -> treat the repo as public
        return None


@st.cache_resource(show_spinner=False)
def load_model():
    weights_path = hf_hub_download(          # cached on disk by huggingface_hub
        repo_id=HF_REPO_ID,
        filename=WEIGHTS_FILENAME,
        token=get_hf_token(),
    )
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_NAME)
    bert = AutoModel.from_pretrained(PRETRAINED_NAME)
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()                              # turn dropout off for inference
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference - same tokenizer settings as training
# ---------------------------------------------------------------------------
def predict(model, tokenizer, text):
    """Returns ([P(Real), P(Fake)], was_truncated)."""
    n_tokens = len(tokenizer(text)["input_ids"])
    enc = tokenizer([text], max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        log_probs = model(enc["input_ids"], enc["attention_mask"])   # model ends in LogSoftmax
    probs = torch.exp(log_probs)[0].tolist()
    return probs, n_tokens > MAX_LENGTH


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Fake News Headline Detector", page_icon="📰")
    st.title("📰 Fake News Headline Detector")
    st.write(
        "This classifier reads a **news headline** and predicts whether it looks like "
        "**Real** or **Fake** news. It is a BERT model fine-tuned on labelled Real and Fake "
        "news headlines."
    )
    st.caption(
        "The model only reads the first 15 tokens (roughly the first dozen words) of the text. "
        "Scores are the model's confidence, not proof that a story is true or false."
    )

    try:
        with st.spinner("Loading the model (the first start can take a minute)..."):
            model, tokenizer = load_model()
    except (EntryNotFoundError, RepositoryNotFoundError):
        st.error(
            f"The model weights could not be found. Make sure `{WEIGHTS_FILENAME}` is uploaded to "
            f"`{HF_REPO_ID}` on Hugging Face. If that repository is private, add your read token "
            "as a secret named `HF_TOKEN`."
        )
        st.stop()
    except Exception as e:
        st.error("The model could not be loaded. Please try again in a moment.")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {e}")
        st.stop()

    text = st.text_area(
        "News headline",
        height=110,
        max_chars=MAX_CHARS,
        placeholder="Trump administration issues new rules on U.S. visa waivers",
    )

    if st.button("Predict", type="primary"):
        text = (text or "").strip()
        if not text:
            st.warning("Please enter a headline first.")
        elif len(text.split()) < MIN_WORDS:
            st.warning(f"Please enter at least {MIN_WORDS} words.")
        else:
            try:
                probs, truncated = predict(model, tokenizer, text)
            except Exception:
                st.error("Sorry, something went wrong while analysing that text. Please try again.")
            else:
                best = max(range(len(CLASS_NAMES)), key=lambda i: probs[i])
                st.subheader(f"Prediction: {CLASS_NAMES[best]}")
                cols = st.columns(len(CLASS_NAMES))
                for col, name, p in zip(cols, CLASS_NAMES, probs):
                    col.metric(f"{name} (model confidence)", f"{p:.1%}")
                if truncated:
                    st.info("Only the beginning of your text was analysed, because the model reads "
                            "headlines (first 15 tokens).")


if __name__ == "__main__":
    main()
