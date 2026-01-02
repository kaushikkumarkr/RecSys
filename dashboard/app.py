import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Configuration
API_URL = "http://api:8000"  # Internal Docker network URL
# For local dev outside docker, user might need localhost:8001, but from inside docker container it is api:8000.
# We will rely on Docker networking.

st.set_page_config(page_title="RecSys Platform", layout="wide")

st.title("🤖 RecSys + Semantic Search Platform")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Search", "Recommendations", "Chat with AI", "System Stats"])

if page == "Search":
    st.header("🔍 Vector Semantic Search")
    query = st.text_input("Enter search query", "finance and technology")
    k = st.slider("Number of results", 1, 20, 5)
    
    if st.button("Search"):
        try:
            res = requests.post(f"{API_URL}/search", params={"query": query, "k": k})
            if res.status_code == 200:
                results = res.json()
                for item in results:
                    with st.expander(f"{item['title']} (Score: {item['similarity']:.2f})"):
                        st.markdown(f"**Category:** {item['category']}")
                        st.text(f"ID: {item['item_id']}")
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

elif page == "Recommendations":
    st.header("🎯 Personalized For You")
    user_id = st.text_input("User ID", "U1")
    
    if st.button("Get Recommendations"):
        try:
            payload = {"user_id": user_id, "k": 10}
            res = requests.post(f"{API_URL}/recommend", json=payload)
            
            if res.status_code == 200:
                recs = res.json()
                if not recs:
                    st.warning("No recommendations found.")
                else:
                    df = pd.DataFrame(recs)
                    st.dataframe(df[['rank', 'title', 'category', 'score', 'reasons', 'strategy']])
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

elif page == "Chat with AI":
    st.header("💬 Chat with Your Data (RAG)")
    st.caption("Ask questions about the articles in the database. Uses semantic search + local LLM.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the news..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(f"{API_URL}/chat", json={"query": prompt, "k": 5}, timeout=120)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer.")
                        sources = data.get("sources", [])
                        
                        st.markdown(answer)
                        if sources:
                            st.caption(f"📚 Sources: {', '.join(sources)}")
                        
                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer + (f"\n\n_Sources: {', '.join(sources)}_" if sources else "")
                        })
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Chat Failed: {e}")

elif page == "System Stats":
    st.header("📊 System Health")
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            st.success("API is Online")
            st.json(res.json())
        else:
            st.error("API is Offline")
    except:
        st.error("API Unreachable")
