import pandas as pd 
import numpy as np 
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


header = st.container()
col1, col2, col3 = st.columns([1,1,1])
dataset = st.container()
recommandation_movie = st.container()



with header:
	st.title("Bienvenue sur notre système de recommandation de films !")
	st.text('Basé sur la base de données IMDB, et des filtres sur la note moyenne (6.3) et le nombre de votes')

with col1:
	st.write("")
with col2:
	st.image('./imdb.svg.png')
with col3:
	st.write("")


with dataset : 
	df_knn = pd.read_csv('./df_final_knn.csv')

with recommandation_movie :
	X = df_knn[['averageRating', 'numVotes','decade', 'action', 'adventure', 'animation', 'biography', 'comedy',
       'crime', 'documentary', 'drama', 'family', 'fantasy', 'film-noir',
       'history', 'horror', 'music', 'musical', 'mystery', 'news', 'romance',
       'sci-fi', 'sport', 'thriller', 'war', 'western', 'Akira Kurosawa', 'Alfred Hitchcock', 'Billy Wilder',
       'Blake Edwards', 'Brian De Palma', 'Christopher Nolan',
       'Clint Eastwood', 'Danny Boyle', 'David Cronenberg', 'David Fincher',
       'David Lynch', 'Ethan Coen', 'Francis Ford Coppola', 'George Cukor',
       'Gus Van Sant', 'Guy Ritchie', 'Hayao Miyazaki', 'Howard Hawks',
       'Ingmar Bergman', 'Joel Coen', 'Joel Schumacher', 'John Carpenter',
       'John Ford', 'John Huston', 'John Landis', 'Lars von Trier',
       'Lasse Hallström', 'Luc Besson', 'Martin Scorsese', 'Michael Haneke',
       'Mike Nichols', 'Neil Jordan', 'Oliver Stone', 'Pedro Almodóvar',
       'Peter Jackson', 'Quentin Tarantino', 'Richard Donner',
       'Richard Linklater', 'Ridley Scott', 'Rob Reiner', 'Robert Rodriguez',
       'Robert Zemeckis', 'Roman Polanski', 'Ron Howard', 'Sam Raimi',
       'Sidney Lumet', 'Stanley Kubrick', 'Stephen Frears',
       'Steven Soderbergh', 'Steven Spielberg', 'Tim Burton', 'Tony Scott',
       'Walter Hill', 'William Wyler', 'Woody Allen']]

	scaler = StandardScaler().fit(X)
	X_scaled = scaler.transform(X)
	df_scaled = pd.DataFrame(data = X_scaled, columns = X.columns, index = X.index)
	df_scaled.insert(0, column = 'title', value = df_knn['title'])
	#df_scaled.insert(1, column = 'url', value = df_knn['poster_url'])
	df_scaled.insert(2, column = 'genres', value = df_knn['genres'])
	df_scaled.insert(3, column = 'réalisateur', value = df_knn['primaryName'])
	weights = pd.Series(np.array([3, 1, 2.5, 2.5, 2.5,2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), index = X.columns)
	st.text("Vous pouvez écrire seulement une partie du titre et choisir ensuite via l'index, ex: harry potter, plutôt qu'harry potter et la coupe de feu ")
	movie = st.text_input("Quel film avez-vous aimé ? (à entrer en minuscule et les accents sont pris en compte)")
	def recommandation2(movie):
		try:
			if df_scaled[df_scaled['title'].str.contains(movie)].shape[0] > 1:
				st.write("Plusieurs films contiennent ce nom, choisissez un index parmis l'un d'eux : ")
				multiple_index = df_knn[df_knn['title'].str.contains(movie)]
				return multiple_index[['title', 'averageRating', 'decade', 'primaryName', 'genres', 'poster_url']]

			else:
				model_film = NearestNeighbors(metric = 'wminkowski', n_neighbors=10, metric_params = {"w": weights}).fit(X_scaled)
				index_reco = model_film.kneighbors(df_scaled.loc[df_scaled['title'].str.contains(movie), X.columns])        
				recommended_movie = df_knn.iloc[df_knn.index.searchsorted(index_reco[1][0][0:9])]
				return recommended_movie[['title', 'averageRating', 'decade', 'primaryName', 'genres']]#, 'poster_url']]
		except ValueError:
			st.warning("Le film choisi n'est pas dans notre base de données, certainement pas assez bien noté ..!")
		except AttributeError:
			return			
			
	try:
		if (df_scaled[df_scaled['title'].str.contains(movie)].shape[0] > 1) is False:	
			if pd.isnull(recommandation2(movie).iloc[1,5]):
				st.write(recommandation2(movie))
			elif pd.notnull(recommandation2(movie).iloc[1,5]):
				st.image(recommandation2(movie).iloc[1,5], width=300)
				st.write(recommandation2(movie))
		else:
			st.write(recommandation2(movie))
	except AttributeError:
		st.write('')

	index = st.text_input("Si nécessaire, quel index avez-vous choisi ?")

	def index_recommandation(index):
			model_film = NearestNeighbors(metric = 'wminkowski', n_neighbors=10, metric_params = {"w": weights}).fit(X_scaled)
			index_reco_2 = model_film.kneighbors(df_scaled.loc[df_scaled.index.isin([index]), X.columns])        
			recommended_movie_2 = df_knn.iloc[df_knn.index.searchsorted(index_reco_2[1][0][0:9])]
			return recommended_movie_2[['title', 'averageRating', 'decade', 'primaryName', 'genres', 'poster_url']]


	if (len(index)>=1) and pd.notnull(index_recommandation(index).iloc[1,5]):
		st.image(index_recommandation(index).iloc[1,5], width=300)
		st.write(index_recommandation(index))
	elif len(index)>=1: 
		st.write(index_recommandation(index))
