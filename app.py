# app.py

import streamlit as st
from brain import store_text, ask_brain, count_memories

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GetSetGo:Second Brain", page_icon="GSG", layout="centered")

st.title("GetSetGo")
st.caption("Store anything. Ask anything. Find it instantly.")

# ── Mode toggle ───────────────────────────────────────────────────────────────
mode = st.radio(
    label="Mode",
    options=["📥 Set", "🔍 Get"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# ── Memory counter ────────────────────────────────────────────────────────────
count = count_memories()
st.caption(f"🗃️ {count} memories stored")

# ── Input box ─────────────────────────────────────────────────────────────────
if mode == "📥 Set":
    st.info("**Set mode** — anything you type gets saved to memory")
    user_input = st.text_area(
        label="What do you want to remember?",
        placeholder='e.g. "Movies to watch: Malli Malli Idhi Rani Roju, Bombay , With Love"',
        height=120
    )
    if st.button(" Go ", use_container_width=True):
        if user_input.strip():
            result = store_text(user_input)
            st.success(result)
        else:
            st.warning("Please type something first!")

elif mode == "🔍 Get":
    st.info("**Get mode** — search your memories with natural language")
    query = st.text_input(
        label="Ask your brain...",
        placeholder='e.g. "What movies did I want to watch?"'
    )
    if st.button(" Go", use_container_width=True):
        if query.strip():
            with st.spinner("Searching your memories..."):
                answer = ask_brain(query)
            st.markdown(answer)
        else:
            st.warning("Please type a question!")