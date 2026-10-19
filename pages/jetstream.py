"""
!!! WARNING !!!
This experiment could use a lot of network bandwidth.
"""
import asyncio
import json
import queue
import threading
import time
import streamlit as st
import torch
import websockets
import pandas as pd

# Bluesky WSS uri for jetstream
# do not use the firehose uri here
JETSTREAM_WSS_URI = "wss://jetstream.us-west.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

# Used to filter the websocket stream so that only finance related posts are analysed by our model. 
FINANCE_KEYWORDS = [
    "$",
    "stock",
    "stocks",
    "shares",
    "market",
    "trading",
    "earnings",
    "revenue",
    "profit",
    "dividend",
    "inflation",
    "interest rate",
    "federal reserve",
    "recession",
    "gdp",
    "bitcoin",
    "btc",
    "ethereum",
    "crypto",
    "bullish",
    "bearish",
    "rally",
    "selloff",
    "volatility",
]

st.set_page_config(
    page_title="Bluesky Jetstream Experiment",
    layout="wide",
)

if "post_queue" not in st.session_state:
  st.session_state.post_queue = queue.Queue(maxsize=100)
if "live_results" not in st.session_state:
  st.session_state.live_results = []
if "sentiment_counts" not in st.session_state:
  st.session_state.sentiment_counts = {}
if "sentiment_batch_number" in st.session_state:
  st.session_state.sentiment_counts = {}
  del st.session_state.sentiment_batch_number
if "jetstream_running" not in st.session_state:
  st.session_state.jetstream_running = False
if "show_live_data" not in st.session_state:
  st.session_state.show_live_data = False

# Asynchronous background loop to listen to the WebSocket
def run_async_loop(msg_queue, stop_event):
  async def listen():
    while not stop_event.is_set():
      try:
        async with websockets.connect(
            JETSTREAM_WSS_URI, subprotocols=["xrpc.v1.json"]
        ) as websocket:
          while not stop_event.is_set():
            try:
              message = await asyncio.wait_for(websocket.recv(), timeout=1)
            except asyncio.TimeoutError:
              continue

            data = json.loads(message)

            # Ensure it's a commit event containing a post
            if data.get("kind") == "commit":
              commit = data.get("commit", {})
              if (
                  commit.get("operation") == "create"
                  and commit.get("collection") == "app.bsky.feed.post"
              ):
                record = commit.get("record", {})
                text = record.get("text", "").lower()

                # Filter text for finance keywords
                if any(kw in text for kw in FINANCE_KEYWORDS):
                  try:
                    msg_queue.put_nowait({
                        "text": record.get("text"),
                        "created_at": record.get(
                            "createdAt", str(time.time())
                        ),
                        "author_did": data.get("did"),
                    })
                  except queue.Full:
                    pass
          await websocket.close()
      except Exception:
        # Reconnect logic or brief pause on connection drop
        await asyncio.sleep(3)

  # Run the async loop inside the thread
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(listen())


def start_websocket_thread(msg_queue, stop_event):
  run_async_loop(msg_queue, stop_event)


# --- Streamlit UI Layout ---
st.title("Monitoring Financial Sentiment on Bluesky")
st.text("This experiment uses Bluesky's Jetstream websocket to read financial posts as they're being posted.")



start_stream = st.button("Start")
stop_stream = st.button("Stop")

if st.session_state.jetstream_running:
  st.success("WebSocket Active...")
else:
  st.info("WebSocket Disconnected.")

# Start Thread Action
if start_stream and not st.session_state.jetstream_running:
  if "model" not in st.session_state or st.session_state.model is None:
    st.error("Please load a model on the main page first.")
  else:
    st.session_state.stop_event = threading.Event()
    st.session_state.thread = threading.Thread(
        target=start_websocket_thread,
        args=(st.session_state.post_queue, st.session_state.stop_event),
        daemon=True,
    )
    st.session_state.thread.start()
    st.session_state.jetstream_running = True
    st.session_state.show_live_data = True
    st.rerun()

# Stop Thread Action
if stop_stream and st.session_state.jetstream_running:
  if "stop_event" in st.session_state:
    st.session_state.stop_event.set()
  if "thread" in st.session_state:
    st.session_state.thread.join(timeout=2)
  st.session_state.jetstream_running = False
  st.session_state.show_live_data = True
  st.rerun()

st.markdown("---")

@st.fragment(run_every="2s")
def render_live_data():
  model = st.session_state.model
  tokenizer = st.session_state.tokenizer

  # Drain items from the thread-safe queue and process them with the model.
  while not st.session_state.post_queue.empty():
    post_data = st.session_state.post_queue.get()

    inputs = tokenizer(
        post_data["text"], return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
      outputs = model(**inputs)
      probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
      pred_class = torch.argmax(probs).item()

    id2label = getattr(model.config, "id2label")
    sentiment = id2label.get(pred_class, f"Class {pred_class}")
    confidence = float(probs[pred_class]) * 100

    st.session_state.live_results.insert(
        0,
        {
            "Time": post_data["created_at"],
            "Post Text": post_data["text"],
            "Predicted Sentiment": sentiment,
            "Confidence": f"{confidence:.1f}%",
        },
    )
    st.session_state.live_results = st.session_state.live_results[:20]

    st.session_state.sentiment_counts[sentiment] = (
        st.session_state.sentiment_counts.get(sentiment, 0) + 1
    )

  processed_post_count = sum(st.session_state.sentiment_counts.values())
  if processed_post_count:
    chart_data = pd.DataFrame(
        {
            "Sentiment": list(st.session_state.sentiment_counts.keys()),
            "Count": list(st.session_state.sentiment_counts.values()),
        }
    ).set_index("Sentiment")
    st.subheader("Live Sentiment Counts")
    st.caption(
        f"Based on {processed_post_count} processed posts "
        f"(updated {pd.Timestamp.now().strftime('%H:%M:%S')})"
    )
    st.bar_chart(chart_data)
  else:
    st.info("Gathering data for visualization...")

  if st.session_state.live_results:
    st.dataframe(
        st.session_state.live_results, width="stretch", hide_index=True
    )
  else:
    st.write("No results yet...")


if st.session_state.show_live_data:
  render_live_data()
else:
  st.info("Start the stream to see the live sentiment breakdown chart.")