import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summerizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decision,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Videos Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Custom Styling
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #0b1220 100%);
            color: white;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.2;
            color: #ffffff;
        }

        .hero p {
            margin-top: 0.4rem;
            color: #cbd5e1;
            font-size: 0.98rem;
        }

        .card {
            padding: 1rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
            color: white;
        }

        .metric-card {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            text-align: center;
        }

        .small-label {
            font-size: 0.8rem;
            color: #94a3b8;
        }

        .value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
        }

        .stTextInput input, .stTextArea textarea, .stSelectbox div, .stFileUploader {
            border-radius: 12px !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        /* Improve tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px 12px 0 0;
            padding: 10px 14px;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.18) !important;
        }

        /* Chat bubbles spacing */
        .chat-msg {
            padding: 0.6rem 0.8rem;
            border-radius: 14px;
            margin-bottom: 0.6rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return path."""
    suffix = Path(uploaded_file.name).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()
    return temp_file.name


def as_markdown(value):
    """Render lists/dicts/strings nicely."""
    if value is None:
        return "_No data found._"
    if isinstance(value, list):
        if not value:
            return "_No data found._"
        return "\n".join([f"- {item}" for item in value])
    if isinstance(value, dict):
        if not value:
            return "_No data found._"
        return "\n".join([f"- **{k}**: {v}" for k, v in value.items()])
    text = str(value).strip()
    return text if text else "_No data found._"


def word_count(text):
    if not text:
        return 0
    return len(str(text).split())


def line_count(value):
    if not value:
        return 0
    if isinstance(value, list):
        return len(value)
    return len([ln for ln in str(value).splitlines() if ln.strip()])


def run_pipeline_ui(source: str, language: str = "english") -> dict:
    """Run the pipeline step by step with progress updates."""
    progress = st.progress(0, text="Initializing...")

    progress.progress(10, text="Processing input...")
    chunks = process_input(source=source)

    progress.progress(30, text="Transcribing audio/video...")
    transcript = transcribe_all(chunks=chunks, language=language)

    progress.progress(50, text="Generating title and summary...")
    title = generate_title(transcript=transcript)
    summary = summarize(transcript=transcript)

    progress.progress(70, text="Extracting action items, decisions, and questions...")
    action_item = extract_action_items(transcript=transcript)
    decision = extract_key_decision(transcript=transcript)
    questions = extract_questions(transcript=transcript)

    progress.progress(90, text="Building chat assistant (RAG)...")
    rag_chain = build_rag_chain(transcript=transcript)

    progress.progress(100, text="Done!")
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_item": action_item,
        "decision": decision,
        "questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------
# Session State
# ----------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "temp_upload_path" not in st.session_state:
    st.session_state.temp_upload_path = None

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🎬 AI Videos Assistant")
    st.caption("Analyze a video, meeting, or podcast and chat with it.")

    st.markdown("---")

    source_url = st.text_input(
        "YouTube URL / Local file path",
        placeholder="Paste a YouTube link or local media file path",
    )

    uploaded_file = st.file_uploader(
        "Or upload a media file",
        type=["mp4", "mp3", "wav", "m4a", "mov", "avi", "mkv"],
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        index=0,
    )

    st.markdown("---")

    analyze_clicked = st.button("🚀 Analyze", use_container_width=True)
    clear_clicked = st.button("🧹 Clear Session", use_container_width=True)

    st.markdown("---")
    st.caption("Tip: upload a file or paste a URL, then click Analyze.")

    if clear_clicked:
        st.session_state.result = None
        st.session_state.messages = []
        if st.session_state.temp_upload_path and os.path.exists(st.session_state.temp_upload_path):
            try:
                os.unlink(st.session_state.temp_upload_path)
            except Exception:
                pass
        st.session_state.temp_upload_path = None
        st.rerun()

# ----------------------------
# Hero Section
# ----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎥 AI Videos Assistant</h1>
        <p>
            Turn videos, meetings, and podcasts into summaries, action items, decisions, and a searchable chat experience.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Main Action
# ----------------------------
if analyze_clicked:
    source = None

    if uploaded_file is not None:
        if st.session_state.temp_upload_path and os.path.exists(st.session_state.temp_upload_path):
            try:
                os.unlink(st.session_state.temp_upload_path)
            except Exception:
                pass
        st.session_state.temp_upload_path = save_uploaded_file(uploaded_file)
        source = st.session_state.temp_upload_path
    elif source_url.strip():
        source = source_url.strip()

    if not source:
        st.warning("Please provide a YouTube URL, local file path, or upload a media file.")
    else:
        try:
            with st.spinner("Running the AI pipeline..."):
                st.session_state.result = run_pipeline_ui(source=source, language=language)

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "I'm ready. Ask me anything about the transcript, decisions, action items, or summary.",
                }
            ]
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ----------------------------
# Result Rendering
# ----------------------------
result = st.session_state.result

if result:
    title = result.get("title", "Untitled")
    transcript = result.get("transcript", "")
    summary = result.get("summary", "")
    action_item = result.get("action_item", "")
    decision = result.get("decision", "")
    questions = result.get("questions", "")
    rag_chain = result.get("rag_chain")

    st.markdown(
        f"""
        <div class="card">
            <h2 style="margin:0 0 0.3rem 0;">{title}</h2>
            <p style="margin:0;color:#cbd5e1;">
                Your AI-powered meeting/video analysis is ready.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Transcript Length</div>
                <div class="value">{len(transcript):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Summary Words</div>
                <div class="value">{word_count(summary):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Action Items</div>
                <div class="value">{line_count(action_item):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Questions</div>
                <div class="value">{line_count(questions):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    tabs = st.tabs(
        ["✨ Summary", "✅ Action Items", "🧠 Key Decision", "❓ Questions", "📜 Transcript", "💬 Chat"]
    )

    with tabs[0]:
        st.markdown("### Summary")
        st.markdown(f"<div class='card'>{as_markdown(summary)}</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### Action Items")
        st.markdown(f"<div class='card'>{as_markdown(action_item)}</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("### Key Decision")
        st.markdown(f"<div class='card'>{as_markdown(decision)}</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("### Questions")
        st.markdown(f"<div class='card'>{as_markdown(questions)}</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("### Full Transcript")
        st.text_area("Transcript", transcript, height=500)
        st.download_button(
            label="⬇️ Download Transcript",
            data=transcript,
            file_name=f"{title[:50].strip().replace(' ', '_')}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tabs[5]:
        st.markdown("### Chat with your meeting")
        st.caption("Ask questions like: 'What were the main decisions?' or 'List the action items.'")

        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Type your question here...")
        if user_question:
            if rag_chain is None:
                st.error("RAG chain is not ready. Please analyze the video first.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = ask_question(rag_chain, user_question)
                            answer_text = str(answer)
                            st.markdown(answer_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer_text}
                            )
                        except Exception as e:
                            st.error(f"Failed to get answer: {e}")

else:
    st.markdown("### Get started")
    st.info("Use the sidebar to paste a URL, enter a local file path, or upload a media file. Then click **Analyze**.")
    st.markdown(
        """
        <div class="card">
            <h4 style="margin-top:0;">What you’ll get</h4>
            <ul style="color:#e2e8f0;">
                <li>Auto-generated title</li>
                <li>Clean summary</li>
                <li>Action items</li>
                <li>Key decisions</li>
                <li>Important questions</li>
                <li>Chat with the transcript</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )