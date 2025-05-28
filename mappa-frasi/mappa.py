import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import io
import spacy
import os

# Se sei in locale, userà 8501. Se sei su Render, userà il valore di $PORT
port = int(os.getenv("PORT", 8501))
st.set_option('server.port', port)

# Config ambientale
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cache per il modello
@st.cache_resource
def load_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Carica modello una sola volta
model = load_model()

# Carica spaCy (ignora errori se manca)
try:
    nlp = spacy.load("it_core_news_sm")
except Exception as e:
    nlp = None
    st.warning("Modulo spaCy non caricato. Assicurati che 'it_core_news_sm' sia installato.")

# Titolo app
st.title("Mappa di similarità tra Frasi")

# Input utente
sentences = []
num = st.number_input("Quante frasi vuoi inserire?", min_value=2, max_value=50, value=5)

for i in range(num):
    sentence = st.text_input(f"Frase {i+1}", "")
    if sentence:
        sentences.append(sentence)

threshold = st.slider("Soglia di similarità per collegare le frasi", 0.1, 0.9, 0.3)

if st.button("Genera mappa"):
    if len(sentences) < 2:
        st.warning("Inserisci almeno 2 frasi.")
    else:
        with st.spinner("Elaborazione in corso..."):
            # Embedding + Similarità
            embeddings = model.encode(sentences)
            similarity_matrix = cosine_similarity(embeddings)

            # Crea grafo
            G = nx.Graph()
            for i, sentence in enumerate(sentences):
                G.add_node(i, label=sentence)

            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = similarity_matrix[i][j]
                    if sim > threshold:
                        G.add_edge(i, j, weight=sim)

            pos = nx.spring_layout(G, seed=42)

            # Visualizza mappa
            plt.figure(figsize=(12, 8))
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=2000)
            labels = {i: s for i, s in enumerate(sentences)}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)

            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            edge_colors = [plt.cm.Blues(w) for w in weights]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=[w * 3 for w in weights])

            plt.title("Mappa delle frasi simili")
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.pyplot(plt)
            plt.close()  # <- Importante per chiudere la figura

            st.success("Mappa generata!")

            # Frasi isolate
            isolated = [n for n in G.nodes if G.degree(n) == 0]
            if isolated:
                st.info(f"Frasi isolate: {[sentences[i] for i in isolated]}")

            st.download_button(
                label="Scarica immagine",
                data=buf.getvalue(),
                file_name="mappa_similarità.png",
                mime="image/png"
            )