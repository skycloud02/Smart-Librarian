import streamlit as st
from retrievers.mock_tfidf import TfidfBookRetriever
# când e gata integrarea, schimbă în sidebar: from retrievers.chroma_store import ChromaRetriever
from chat_core import ask_chat
from retrievers.chroma_store import build_index, ChromaRetriever

st.set_page_config(page_title="Smart Librarian – Chat", page_icon=" ")
st.sidebar.title("Setări Chat")
backend = st.sidebar.selectbox("Retriever", ["TF-IDF (local)", "Chroma (OpenAI)"])


#buton optional de (re)indexare
if backend == "Chroma (OpenAI)":
    if st.sidebar.button("Construiește / Reconstruiește index"):
        n = build_index()
        st.sidebar.success(f"Index updatat cu {n} cărți.")

#initializeaza retrieverul selectat
if backend == "TF-IDF (local)":
    retriever = TfidfBookRetriever(csv_path="data/books.csv")
else:
    retriever = ChromaRetriever(k=10)

# # Inițializează retrieverul selectat
# if backend == "TF-IDF (local)":
# 	retriever = TfidfBookRetriever(csv_path="data/books.csv")
# else:
# 	# Placeholder: când e gata, înlocuiești cu ChromaRetriever(...)
# 	st.sidebar.info("Chroma va fi disponibil când e gata embeddings + index.")
# 	retriever = TfidfBookRetriever(csv_path="data/books.csv")

st.title("Smart Librarian – Chat")

if "messages" not in st.session_state:
	st.session_state.messages = [
		{
			"role": "assistant",
			"content": "Salut! Spune-mi ce fel de carte cauți (genuri, teme, vibe).",
		}
	]

# afișare istoric
for m in st.session_state.messages:
	with st.chat_message(m["role"]):
		st.markdown(m["content"])

# input
if prompt := st.chat_input("Ex: Vreau o carte despre prietenie și magie"):
	st.session_state.messages.append({"role": "user", "content": prompt})
	history = [
		(m["role"], m["content"])
		for m in st.session_state.messages
		if m["role"] in {"user", "assistant"}
	]
	with st.chat_message("assistant"):
		with st.spinner("Caut sugestii potrivite..."):
			reply = ask_chat(prompt, retriever=retriever, history=history)
			st.markdown(reply)
	st.session_state.messages.append({"role": "assistant", "content": reply})