import os
import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# API
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ----------------------------------------
# 1. scraping do TMDB
# ----------------------------------------
@st.cache_data(ttl=86400)  # Cache válido por 24 horas
def fetch_tmdb_movies(pages=10):
    movies = []
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=pt-BR&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            movies.extend(response.json().get("results", []))
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao acessar a API: {e}")
            break
    return movies

@st.cache_data(ttl=86400)
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=pt-BR"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return {genre["id"]: genre["name"] for genre in response.json().get("genres", [])}
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar gêneros: {e}")
        return {}

# ----------------------------------------
# 2. Pré-processamento
# ----------------------------------------
def preprocess_data(movies, genre_map):
    df = pd.DataFrame([{
        "id": m["id"],
        "title": f"{m['title']} ({m['release_date'][:4]})" if m.get("release_date") else m["title"],
        "genres": ", ".join([genre_map.get(g, "Desconhecido") for g in m.get("genre_ids", [])]),
        "overview": m.get("overview", ""),
        "release_date": m.get("release_date", ""),
        "vote_average": m.get("vote_average", "N/A")
    } for m in movies if m.get("title")])

    # Combinar features para o modelo
    df["features"] = df["title"] + " " + df["genres"] + " " + df["overview"]
    return df

# ----------------------------------------
# 3. modelo recomendando
# ----------------------------------------
class TMDBRecommender:
    def __init__(self, df):
        self.df = df
        self.tfidf = TfidfVectorizer()
        self.cosine_sim = None

    def train(self):
        tfidf_matrix = self.tfidf.fit_transform(self.df["features"])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, movie_title, top_n=5):
        idx = self.df[self.df["title"] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        # Remover duplicatas (usando um set para garantir filmes únicos)
        seen = set()
        recommendations = []
        for i in sim_scores:
            movie = self.df.iloc[i[0]]
            if movie["id"] not in seen:
                seen.add(movie["id"])
                recommendations.append(movie)
        
        return recommendations

# ----------------------------------------
# 4. Personalização
# ----------------------------------------
def main():
    st.set_page_config("CineMatch 🎬", layout="wide")

    # Carregar dados
    with st.spinner("Carregando dados do TMDB..."):
        movies_data = fetch_tmdb_movies()
        genre_map = fetch_genres()
        df = preprocess_data(movies_data, genre_map)
        recommender = TMDBRecommender(df)
        recommender.train()

    # Sidebar
    st.sidebar.title("👇")
    selected_movie = st.sidebar.selectbox("Escolha um filme:", df["title"].tolist())

    # Página principal
    st.title("CineMatch - Seu Curador de Filmes Pessoal")
    
    # Detalhes do filme selecionado
    selected_details = next((m for m in movies_data if m["id"] == df[df["title"] == selected_movie]["id"].values[0]), None)
    if selected_details:
        col1, col2 = st.columns([1, 3])
        with col1:
            poster_url = f"https://image.tmdb.org/t/p/w500{selected_details['poster_path']}" if selected_details.get("poster_path") else None
            if poster_url:
                st.image(poster_url, caption=selected_movie, use_container_width=True)
            else:
                st.warning("Poster não disponível")
        with col2:
            st.subheader("Sinopse")
            st.write(selected_details.get("overview", "Sinopse não disponível"))
            st.markdown(f"**Avaliação TMDB:** ⭐ {selected_details.get('vote_average', 'N/A')}/10")
            st.markdown(f"**Data de Lançamento:** {selected_details.get('release_date', 'Desconhecida')}")
    
    # Recomendações
    st.divider()
    if st.button("Gerar Recomendações"):
        with st.spinner("Gerando recomendações..."):
            recommendations = recommender.get_recommendations(selected_movie)

        st.subheader("🎯 Recomendações Para Você")
        cols = st.columns(3)
        for idx, row in enumerate(recommendations):
            with cols[idx % 3]:
                movie_data = next((m for m in movies_data if m["id"] == row["id"]), {})
                with st.expander(row["title"], expanded=True):
                    if movie_data.get("poster_path"):
                        st.image(f"https://image.tmdb.org/t/p/w300{movie_data['poster_path']}", use_container_width=True)
                    st.caption(f"Gêneros: {row['genres']}")
                    st.write(movie_data.get("overview", "Sem descrição disponível"))
                    rating = movie_data.get("vote_average", "N/A")  # Pega a nota média
                    st.caption(f"Avaliação TMDB: ⭐ {rating}/10") 

if __name__ == "__main__":
    main()
