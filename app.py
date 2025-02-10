import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pickle
import os

# ----------------------------------------
# 1. Pré-processamento dos Dados
# ----------------------------------------
def load_and_preprocess_data():
    movies = pd.read_csv('data/movies.csv')    
    ratings = pd.read_csv('data/ratings.csv')
    
    # Converter IDs para inteiros
    movies['movieId'] = movies['movieId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['userId'] = ratings['userId'].astype(int)
    
    # Processamento dos gêneros
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    ratings = ratings[ratings['movieId'].isin(popular_movies)]
    
    return movies, ratings

# ----------------------------------------
# 2. Funções de Visualização (EDA) - CORRIGIDO
# ----------------------------------------
def generate_similarity_network(movies, cosine_sim, selected_movie, top_n=5):
    G = nx.Graph()
    idx = movies[movies['title'] == selected_movie].index[0]
    
    # Adicionar nós e conexões
    G.add_node(selected_movie, size=15, color='#FF6B6B')
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    for i, score in sim_scores:
        movie_title = movies.iloc[i]['title']
        G.add_node(movie_title, size=10, color='#4B8DFF')
        G.add_edge(selected_movie, movie_title, weight=score*10)  # <-- Correção aqui

    # Layout do grafo
    pos = nx.spring_layout(G, seed=42)
    
    # Plotar com Plotly
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines'),
            go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                       marker=dict(color=node_colors, size=node_sizes), textposition="top center")
        ],
        layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=30))
    )
    return fig

def perform_eda(ratings, movies, cosine_sim):
    sns.set_style("whitegrid")
    
    # Word Cloud
    st.subheader("🌍 Nuvem de Gêneros")
    genres_text = ' '.join(movies['genres'].explode())
    wordcloud = WordCloud(width=800, height=400).generate(genres_text)
    fig = px.imshow(wordcloud.to_array())
    st.plotly_chart(fig)

    # Rede de Similaridade
    st.subheader("🔗 Rede de Conexões entre Filmes")
    selected_movie = st.selectbox("Selecione um filme:", movies['title'])
    if st.button("Gerar Rede"):
        network_fig = generate_similarity_network(movies, cosine_sim, selected_movie)
        st.plotly_chart(network_fig, use_container_width=True)

    # Gráficos restantes
    st.subheader("📊 Distribuição de Avaliações")
    plt.figure(figsize=(10,5))
    sns.histplot(ratings['rating'], bins=5, kde=False)
    st.pyplot(plt)
    plt.clf()

# ----------------------------------------
# 3. Sistema de Recomendação
# ----------------------------------------
class RecommendationSystem:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.svd_model = None

    def train_content_based(self):
        self.movies['genres_str'] = self.movies['genres'].apply(lambda x: ' '.join(x))
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres_str'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def train_collaborative(self):
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(self.ratings[['userId','movieId','rating']], reader)
        trainset = data.build_full_trainset()
        self.svd_model = SVD(n_factors=50, random_state=42)
        self.svd_model.fit(trainset)

    def get_content_recommendations(self, title):
        idx = self.movies[self.movies['title'] == title].index[0]
        sim_scores = sorted(enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
        return self.movies.iloc[[i[0] for i in sim_scores]]['title']

    def get_collaborative_recommendations(self, user_id):
        user_movies = self.ratings[self.ratings['userId'] == user_id]['movieId']
        all_movies = self.movies['movieId'].unique()
        unseen = [m for m in all_movies if m not in user_movies]
        
        if not unseen:
            return ["Nenhuma recomendação disponível. O usuário já avaliou todos os filmes! 🎬"]
        
        predictions = [self.svd_model.predict(user_id, m) for m in unseen]
        top_movies_ids = [p.iid for p in sorted(predictions, key=lambda x: x.est, reverse=True)[:10]]
        
        # Verificar se os IDs recomendados existem no dataset
        valid_ids = self.movies[self.movies['movieId'].isin(top_movies_ids)]['movieId'].tolist()
        if not valid_ids:
            return ["Não encontramos recomendações relevantes. 😕"]
        
        return self.movies[self.movies['movieId'].isin(valid_ids)]['title'].tolist()

# ----------------------------------------
# 4. Interface Streamlit
# ----------------------------------------
def main():
    st.set_page_config(page_title="Movie Recommender Pro", layout="wide")
    movies, ratings = load_and_preprocess_data()
    recommender = RecommendationSystem(movies, ratings)

    @st.cache_resource
    def train_models():
        recommender.train_content_based()
        recommender.train_collaborative()
        return recommender
    
    recommender = train_models()

    # Sidebar
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Navegar", ["Home", "Recomendações", "Análise"])

    # Páginas
    if page == "Home":
        st.title("🎥 Sistema de Recomendação de Filmes")
        st.image("https://i.imgur.com/YbL9oZa.jpg", use_column_width=True)
        st.markdown("""
        **Como usar:**
        1. 🗂️ Navegue para a página de **Recomendações**
        2. 🎯 Escolha entre recomendações baseadas em conteúdo ou colaborativas
        3. 📊 Explore insights na página de **Análise**
        """)

    elif page == "Recomendações":
        st.title("🎯 Obter Recomendações")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseada em Conteúdo")
            movie = st.selectbox("Selecione um filme:", movies['title'])
            if st.button("🔍 Buscar Similares"):
                recs = recommender.get_content_recommendations(movie)
                st.write("### Filmes Recomendados:")
                for i, title in enumerate(recs, 1):
                    st.write(f"{i}. {title}")

        with col2:
            st.subheader("Colaborativa")
            user_id = st.number_input("ID do Usuário (1-610)", 1, 610, 1)
            if st.button("✨ Gerar Recomendações"):
                recs = recommender.get_collaborative_recommendations(user_id)
                st.write(f"### Para o usuário {user_id}:")
                for i, title in enumerate(recs, 1):
                    st.write(f"{i}. {title}")

    elif page == "Análise":
        st.title("📈 Análise dos Dados")
        perform_eda(ratings, movies, recommender.cosine_sim)

if __name__ == "__main__":
    main()