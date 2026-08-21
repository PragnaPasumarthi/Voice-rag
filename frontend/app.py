import os
import sys
import streamlit as st
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")


def display_results(data):
    status = data.get("status", "unknown")
    if status == "success":
        st.success(f"**Answer:** {data.get('answer', '')}")
    elif status == "guardrail_blocked":
        st.warning(f"**Blocked:** {data.get('error', data.get('answer', 'Query blocked by guardrails'))}")
    elif status == "no_retrieval":
        st.warning(f"**No match:** {data.get('answer', 'No relevant documents found')}")
    else:
        st.error(f"**Error:** {data.get('error', 'Unknown error')}")

    latency = data.get("latency_ms", 0)
    st.metric("Latency", f"{latency:.1f} ms")

    contexts = data.get("contexts", [])
    scores = data.get("scores", [])
    if contexts:
        with st.expander("Retrieved Contexts"):
            for i, (ctx, score) in enumerate(zip(contexts, scores)):
                st.markdown(f"**Context {i+1}** (score: {score:.3f})")
                st.text(ctx[:500])

    guardrail = data.get("guardrail")
    if guardrail:
        with st.expander("Guardrail Details"):
            st.json(guardrail)


def main():
    st.set_page_config(
        page_title="Voice-Enabled RAG | HH Goa 2026",
        page_icon="microphone",
        layout="wide",
    )

    st.title("Voice-Enabled RAG")
    st.subheader("HH Goa 2026 - Task 2")

    with st.sidebar:
        st.header("Settings")
        language = st.selectbox(
            "Language",
            ["en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa"],
            index=0,
        )
        top_k = st.slider("Top K results", 1, 20, 8)
        st.divider()
        st.markdown("**LLM**: NVIDIA NIM (free)")
        st.markdown("**STT**: Whisper (local, free)")
        st.markdown("**Pipeline**: Voice -> STT -> Chunking -> VectorDB -> LLM -> Guardrails")
        st.markdown("**Strategies**: Fixed, Sliding Window, Sentence, Semantic, Metadata-Aware")

    tab_text, tab_voice, tab_analytics = st.tabs(
        ["Text Query", "Voice Query", "Analytics"]
    )

    with tab_text:
        st.markdown("### Ask a question")
        query = st.text_input(
            "Enter your question:", key="text_query",
            placeholder="What is machine learning?"
        )

        if st.button("Search", key="text_search") and query:
            with st.spinner("Processing..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/query",
                        json={"query": query, "language": language, "top_k": top_k},
                        timeout=60,
                    )
                    data = resp.json()
                    display_results(data)
                except httpx.ConnectError:
                    st.error("Cannot connect to API. Run: python -m backend.main")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_voice:
        st.markdown("### Speak your question")
        audio_file = st.file_uploader(
            "Upload audio (WAV/MP3)", type=["wav", "mp3", "ogg", "webm"],
            key="voice_upload"
        )

        if audio_file:
            st.audio(audio_file, format="audio/wav")
            if st.button("Transcribe & Search", key="voice_search"):
                with st.spinner("Transcribing and searching..."):
                    try:
                        files = {"audio": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        data = {"language": language, "top_k": str(top_k)}
                        resp = httpx.post(
                            f"{API_BASE}/voice-query", files=files, data=data, timeout=60
                        )
                        result = resp.json()

                        if result.get("transcribed_text"):
                            st.info(f"Transcribed: **{result['transcribed_text']}**")

                        display_results(result)
                    except httpx.ConnectError:
                        st.error("Cannot connect to API server.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_analytics:
        st.markdown("### Latency Analytics")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Benchmark (50 queries)", key="benchmark"):
                with st.spinner("Running benchmark..."):
                    try:
                        resp = httpx.post(
                            f"{API_BASE}/analytics/benchmark?queries=50", timeout=120
                        )
                        data = resp.json()
                        stats = data.get("stats", {})

                        if stats:
                            st.success(f"Benchmark complete! ({stats.get('count', 0)} queries)")

                            cols = st.columns(4)
                            cols[0].metric("P50", f"{stats.get('p50', 0):.1f} ms")
                            cols[1].metric("P70", f"{stats.get('p70', 0):.1f} ms")
                            cols[2].metric("P90", f"{stats.get('p90', 0):.1f} ms")
                            cols[3].metric("P100", f"{stats.get('p100', 0):.1f} ms")

                            st.json(stats)
                    except Exception as e:
                        st.error(f"Benchmark error: {e}")

        with col2:
            if st.button("View Current Stats", key="view_stats"):
                try:
                    resp = httpx.get(f"{API_BASE}/analytics/report", timeout=10)
                    report = resp.json().get("report", "No data")
                    st.code(report)
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
