"""
Fake News Headline Detector - Streamlit inference app.

Inference only: the BERT model is never trained here. The trained weights
(c2_new_model_weights.pt) are downloaded from the Hugging Face Hub and loaded
into the same BERT_Arch used in the notebook. Paste a headline or an article
link (the headline is extracted with the same scraping code as the notebook).
"""
import html
import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
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
MIN_WORDS = 3
MAX_CHARS = 1000

TIMEOUT, MAX_BYTES, MAX_REDIRECTS = 10, 2_000_000, 4
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDemo/1.0)"}


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
# Link scraping - same logic as the notebook's Gradio section
# ---------------------------------------------------------------------------
def looks_like_url(text):
    return bool(re.match(r"^([a-z][a-z0-9+.-]*://|www\.)\S+$", text.strip(), re.I))


def check_url(url):
    """Allow only http/https and block localhost / private / internal addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Please enter a valid http:// or https:// URL.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError("Could not find that website. Check the address.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("That address points to a private or internal network, so it is blocked.")


def fetch_html(url):
    """Download a page, following redirects manually so every destination is re-validated."""
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=False)
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("Location"); r.close()
            if not loc:
                raise ValueError("The site sent a broken redirect.")
            url = urljoin(url, loc)
            continue
        if r.status_code != 200:
            r.close()
            raise ValueError(f"The page returned HTTP {r.status_code}. It may block automated access "
                             "or need a login. You can paste the headline instead.")
        if "html" not in r.headers.get("Content-Type", "").lower():
            r.close()
            raise ValueError("That link is not a web page. You can paste the headline instead.")
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > MAX_BYTES:      # keep only the first 2 MB
                break
        r.close()
        return data
    raise ValueError("Too many redirects.")


def clean_headline(t):
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+[-–—|]\s+[^-–—|]{2,30}$", "", t).strip()      # drop a trailing " - Site Name"


def extract_article(url):
    """Returns (headline, body_preview) from a news page."""
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    headline = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content", "").strip():
        headline = og["content"]
    elif soup.find("h1") and soup.find("h1").get_text(strip=True):
        headline = soup.find("h1").get_text(" ", strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        headline = soup.title.get_text(strip=True)
    headline = clean_headline(headline)
    if not headline:
        raise ValueError("Could not find a headline on that page. You can paste it instead.")
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    preview = " ".join(p for p in paras if len(p) >= 60)[:600]
    return headline, preview


def pick_headline(text):
    """The model expects a headline: use the first line (or first sentence if it is a long block)."""
    h = [l.strip() for l in text.splitlines() if l.strip()][0]
    if len(h.split()) > 30:
        h = re.split(r"(?<=[.!?])\s+", h)[0]
    return h


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
<style>
.block-container {max-width: 760px; padding-top: 2.2rem;}
.hero {text-align:center; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {opacity: .75; margin: 0;}
.card {border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
       border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1rem;}
.label {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; opacity: .6; margin-bottom: .25rem;}
.headline {font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; line-height: 1.4;}
.badge {display:inline-block; font-size: 1.5rem; font-weight: 700; padding: .3rem 1rem; border-radius: 999px;}
.badge.real {background: rgba(37,99,235,.15); color: #3b82f6; border: 1px solid rgba(59,130,246,.45);}
.badge.fake {background: rgba(220,38,38,.15); color: #ef4444; border: 1px solid rgba(239,68,68,.45);}
.bar-row {display:flex; align-items:center; gap:.8rem; margin-top:.7rem;}
.bar-name {width: 3.2rem; font-weight: 600;}
.bar-track {flex:1; height: 12px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-fill.real {background: #3b82f6;}
.bar-fill.fake {background: #ef4444;}
.bar-pct {width: 4rem; text-align: right; font-variant-numeric: tabular-nums;}
.note {font-size: .85rem; opacity: .7; margin-top: 1rem;}
</style>
"""


def result_card(headline, probs, notes):
    best = max(range(len(CLASS_NAMES)), key=lambda i: probs[i])
    cls = CLASS_NAMES[best].lower()
    bars = "".join(
        f'<div class="bar-row"><div class="bar-name">{n}</div>'
        f'<div class="bar-track"><div class="bar-fill {n.lower()}" style="width:{p*100:.1f}%"></div></div>'
        f'<div class="bar-pct">{p:.1%}</div></div>'
        for n, p in zip(CLASS_NAMES, probs)
    )
    notes_html = "".join(f'<div class="note">{html.escape(n)}</div>' for n in notes)
    st.markdown(
        f'<div class="card">'
        f'<div class="label">Headline analysed</div><div class="headline">{html.escape(headline)}</div>'
        f'<div class="label">Prediction</div><span class="badge {cls}">{CLASS_NAMES[best]}</span>'
        f'<div class="label" style="margin-top:1.1rem">Model confidence</div>{bars}{notes_html}</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>📰 Fake News Detector</h1>'
        "<p>Paste a news headline or an article link and get a Real / Fake prediction "
        "from a fine-tuned BERT model.</p></div>",
        unsafe_allow_html=True,
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

    user_input = st.text_area(
        "Headline or article link",
        height=110,
        max_chars=MAX_CHARS,
        placeholder="Trump administration issues new rules on U.S. visa waivers\n\nor  https://example.com/news/article",
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        text = (user_input or "").strip()
        if not text:
            st.warning("Please paste a headline or an article link first.")
            st.stop()

        notes, preview = [], ""
        if looks_like_url(text):
            url = text if "://" in text else "https://" + text
            try:
                with st.spinner("Reading the page..."):
                    headline, preview = extract_article(url)
            except ValueError as e:
                st.warning(str(e))
                st.stop()
            except requests.exceptions.Timeout:
                st.warning("The page took too long to respond. Try again or paste the headline.")
                st.stop()
            except requests.exceptions.RequestException:
                st.warning("Could not download that page. Try again or paste the headline.")
                st.stop()
            notes.append("Headline extracted from the page. The article body is not used by the model.")
        else:
            headline = pick_headline(text)
            if len(text.split()) > len(headline.split()):
                notes.append("Only the first line / sentence was analysed, because the model works on headlines.")

        if len(headline.split()) < MIN_WORDS:
            st.warning(f"Not enough text to analyse (need at least {MIN_WORDS} words).")
            st.stop()

        try:
            probs, truncated = predict(model, tokenizer, headline)
        except Exception:
            st.error("Sorry, something went wrong while analysing that text. Please try again.")
            st.stop()

        if truncated:
            notes.append("The model reads only the first 15 tokens (roughly the first dozen words) of a headline.")
        notes.append("Scores are the model's confidence, not proof that a story is true or false.")
        result_card(headline, probs, notes)

        if preview:
            with st.expander("Article preview (for reference, not used by the model)"):
                st.write(preview)


if __name__ == "__main__":
    main()
