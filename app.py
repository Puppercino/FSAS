import os
import gc
import json
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="Financial Sentiment Analyser",
    layout="wide",
)

st.title("Financial Sentiment Analyser")

# Define models directory
MODELS_DIR = "./models"


# Helper function to discover local models
def get_local_models():
  if not os.path.exists(MODELS_DIR):
    return []
  return [
      d
      for d in os.listdir(MODELS_DIR)
      if os.path.isdir(os.path.join(MODELS_DIR, d))
  ]


available_models = get_local_models()

# Session State
if "loaded_model_name" not in st.session_state:
  st.session_state.loaded_model_name = None
if "model" not in st.session_state:
  st.session_state.model = None
if "tokenizer" not in st.session_state:
  st.session_state.tokenizer = None

def read_model_info(model_id):
  """Reads a model's manifest to be shown in the UI."""
  manifest_path = os.path.join(MODELS_DIR, model_id, "manifest.json")
  manifest = {}
  try:
    with open(manifest_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
      manifest["name"] = data["name"]
      manifest["description"] = data["description"]
  except:
    print("The model's manifest may be missing or invalid. The model will still work, but detailed information about the model won't be available.")
    manifest["name"] = model_id
    manifest["description"] = "No description available."
  
  return manifest

def load_model_and_tokenizer(model_path):
  """Load a model and tokenizer."""
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  model.eval()  # Set to evaluation mode
  return model, tokenizer


def unload_model():
  """Release the current model and tokenizer from RAM."""
  st.session_state.model = None
  st.session_state.tokenizer = None
  st.session_state.loaded_model_name = None
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


# Model Selection Menu
selected_model_folder = st.selectbox(
    "Model",
    options=available_models,
    disabled=not available_models,
)

with st.container():
  if selected_model_folder:
    selected_model_manifest = read_model_info(selected_model_folder)
    st.markdown(
        f"### {selected_model_manifest['name']}\n"
        f"{selected_model_manifest['description']}"
    )

    if st.session_state.loaded_model_name != selected_model_folder:
      model_path = os.path.join(MODELS_DIR, selected_model_folder)
      # avoid wasting memory on unused models
      unload_model()
      try:
        with st.spinner(f"Loading {selected_model_folder}..."):
          st.session_state.model, st.session_state.tokenizer = (
              load_model_and_tokenizer(model_path)
          )
          st.session_state.loaded_model_name = selected_model_folder
      except Exception as e:
        st.error(f"Failed to load model: {e}")
  elif st.session_state.model is not None:
    unload_model()

  if st.session_state.loaded_model_name:
    st.success(f"Loaded model: `{st.session_state.loaded_model_name}`")
  else:
    st.warning("No model currently loaded.")

st.markdown("---")

# Text Input
text_input = st.text_area(
    "Text Input",
    placeholder="Enter financial text or news headline to analyse...",
    height=100,
)

analyse_clicked = st.button("Analyse")

# Sentiment analysis results 
st.markdown("### Sentiment Analysis Result")
results_container = st.container(border=True)

if analyse_clicked:
  if not st.session_state.model:
    with results_container:
      st.error("Please select a model first.")
  elif not text_input.strip():
    with results_container:
      st.warning("Please enter some text to analyse.")
  else:
    with results_container:
      with st.spinner("Model analysing..."):
        # Perform inference
        inputs = st.session_state.tokenizer(
            text_input, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
          outputs = st.session_state.model(**inputs)
          probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Use id2label to get the emotion labels from our datasets
        id2label = getattr(
            st.session_state.model.config,
            "id2label"
        )

        st.write("**Prediction Probabilities:**")
        for idx, prob in enumerate(probs[0].tolist()):
          label_name = id2label.get(idx, f"Class {idx}")
          st.progress(
              float(prob), text=f"**{label_name}**: {prob * 100:.2f}%"
          )
else:
  with results_container:
    st.text("Sentiment analysis results will appear here when you click Analyse.")