# -*- coding: utf-8 -*-

# -- Sheet --

import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.express as px 

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import plotipy 
import plotly.graph_objects as go
import songrecommendations

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from math import sqrt

from PIL import Image
import visualize

from sklearn import datasets
from yellowbrick.target import FeatureCorrelation

SPOTIPY_CLIENT_ID="53f107ba55324537ab90a576e29d2287"
SPOTIPY_CLIENT_SECRET="118da4a35c644a5eb8b0c4aefe0f1dd6"


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="53f107ba55324537ab90a576e29d2287", client_secret="118da4a35c644a5eb8b0c4aefe0f1dd6"))




st.image('https://images.prismic.io/soundcharts/727545d02420e55c5c6a376f633a1f02ebc59dc5_mapspot2.png?auto=compress,format',caption = None,width= 650)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Music Recommendation Engine with Spotipy')

st.sidebar.title('Spotify Analysis Dashboard')


search_choices = ['Song/Track', 'Artist', 'Album']
search_selected = st.sidebar.selectbox("Your search choice please: ", search_choices)

search_keyword = st.text_input(search_selected)
button_clicked = st.button("Search")


search_results = []
tracks = []
artists = []
albums = []
if search_keyword is not None and len(str(search_keyword)) > 0:
    if search_selected == 'Song/Track':
        st.write("Start song/track search")
        tracks = sp.search(q='track:'+ search_keyword,type='track', limit=20)
        tracks_list = tracks['tracks']['items']
        if len(tracks_list) > 0:
            for track in tracks_list:
                #st.write(track['name'] + " - By - " + track['artists'][0]['name'])
                search_results.append(track['name'] + " - By - " + track['artists'][0]['name'])
        
    elif search_selected == 'Artist':
        st.write("Start artist search")
        artists = sp.search(q='artist:'+ search_keyword,type='artist', limit=20)
        artists_list = artists['artists']['items']
        if len(artists_list) > 0:
            for artist in artists_list:
                # st.write(artist['name'])
                search_results.append(artist['name'])
        
    if search_selected == 'Album':
        st.write("Start album search")
        albums = sp.search(q='album:'+ search_keyword,type='album', limit=20)
        albums_list = albums['albums']['items']
        if len(albums_list) > 0:
            for album in albums_list:
                # st.write(album['name'] + " - By - " + album['artists'][0]['name'])
                # print("Album ID: " + album['id'] + " / Artist ID - " + album['artists'][0]['id'])
                search_results.append(album['name'] + " - By - " + album['artists'][0]['name'])
    
            

selected_album = None
selected_artist = None
selected_track = None
if search_selected == 'Song/Track':
    selected_track = st.selectbox("Select your song/track: ", search_results)
elif search_selected == 'Artist':
    selected_artist = st.selectbox("Select your artist: ", search_results)
elif search_selected == 'Album':
    selected_album = st.selectbox("Select your album: ", search_results)


if selected_track is not None and len(tracks) > 0:
    tracks_list = tracks['tracks']['items']
    track_id = None
    if len(tracks_list) > 0:
        for track in tracks_list:
            str_temp = track['name'] + " - By - " + track['artists'][0]['name']
            if str_temp == selected_track:
                track_id = track['id']
                track_album = track['album']['name']
                img_album = track['album']['images'][1]['url']
    selected_track_choice=None
    if track_id is not None:
        track_choices = ['Song Features', 'Similar Songs Recommendation','View Dataset & Visualizations','Playlist Analytics']
        selected_track_choice = st.sidebar.selectbox('Please select track choice: ', track_choices)  
        if selected_track_choice=='Song Features':
            track_features= sp.audio_features(track_id)
            df = pd.DataFrame(track_features,index=[0])
            #df_pop=pd.DataFrame(track_features,index=[0])
            df_features= df.loc[: ,['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']]
            st.dataframe(df_features)
            plotipy.feature_plot(df_features)
            
        elif selected_track_choice == 'Similar Songs Recommendation':
            token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
            similar_songs_json = songrecommendations.get_track_recommendations(track_id, token)
            recommendation_list = similar_songs_json['tracks']
            recommendation_list_df = pd.DataFrame(recommendation_list)
            # st.dataframe(recommendation_list_df)
            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
            st.dataframe(recommendation_df)
            # st.write("Recommendations....")
            songrecommendations.song_recommendation_vis(recommendation_df)
        elif selected_track_choice == 'View Dataset & Visualizations':
            data = pd.read_csv("data.csv")
            genre_data = pd.read_csv('data_w_genres.csv')
            year_data = pd.read_csv('data_by_year.csv')
            st.dataframe(data)

            sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
            fig = px.line(year_data, x='year', y=sound_features)
            st.write(fig)
            top10_genres = genre_data.nlargest(10, 'popularity')
            top10_genres = genre_data.nlargest(10, 'popularity')

            fig1 = px.bar(top10_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
            st.write(fig1)

        elif selected_track_choice == 'Playlist Analytics':

            columns = ['name', 'artist', 'track_URI', 'playlist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'tempo', 'valence']

            def main():
                st.markdown("Playlist Analytics")
                st.markdown("Paste your playlist URL to run the algorithm.")
                num_playlists = st.sidebar.number_input('How many playlists would you like to cluster?', 1, 5, 2)
                playlists = playlist_user_input(num_playlists)
                if st.sidebar.button("Run Algorithm") :
                    print(playlists)
                    # acquire the data via Spotify API
                    df = concatenate_playlists(playlists)
                    if df is None:
                        st.warning("One of your playlist URIs was not entered properly")
                        st.stop()
                    else:
                        # dataframe for inspection and exploration
                        st.write(df)

                        # implement k-means clustering with PCA
                        clustered_df, n_clusters = kmeans(df)
                        
                        # make radar chart to help understand the cluster differences
                        cluster_labels = clustered_df['Cluster']
                        orig = clustered_df.drop(columns=['Cluster', "Component 1", "Component 2"])
                        orig.insert(4, "cluster", cluster_labels)
                        norm_df = make_normalized_df(orig, 5)
                        fig, maxes = make_radar_chart(norm_df, n_clusters)
                        st.write(fig)

                        # interactive visualizations of clusters on 2D plane
                        range_ = get_color_range(n_clusters)
                        visualize_clusters(clustered_df, n_clusters, range_)

                        # within-cluster exploration
                        explore_df = orig.copy()
                        keys = sorted(list(explore_df["cluster"].unique()))
                        cluster = st.selectbox("Choose a cluster to preview", keys, index=0)
                        preview_df = preview_cluster_playlist(explore_df, cluster)
                        st.write(preview_df[preview_df.columns[:5]])
                        x_axis = list(preview_df['name'])
                        y_axis = st.selectbox("Choose a variable for the y-axis", list(preview_df.columns)[5:], index=maxes[cluster])
                        visualize_data(preview_df, x_axis, y_axis, n_clusters, range_)
                else:
                    pass

            def playlist_user_input(num_playlists):
                playlists = []
                defaults = ["spotify:playlist:37i9dQZF1DX9UhtB5CtZ7e", "spotify:playlist:37i9dQZF1DWSP55jZj2ES3",
                "spotify:playlist:37i9dQZF1DX4OzrY981I1W",
                "spotify:playlist:37i9dQZF1DX8dTWjpijlub",
                "spotify:playlist:37i9dQZF1DWUE76cNNotSg"
                ]
                for i in range(num_playlists):
                    playlists.append(st.sidebar.text_input("Playlist URI " + str(i+1), defaults[i]))
                return playlists

            def concatenate_playlists(playlists):
                global columns
                print("concatenate playlists")
                df = pd.DataFrame(columns=columns)
                if all(playlists):
                    for playlist_uri in playlists:
                        df = pd.concat([df, get_features_for_playlist(os.environ.get('USERNAME'), playlist_uri)], ignore_index=True, axis=0)
                    return df
                else:
                    return None

            # Get Spotipy credentials from config
            def load_config():
                stream = open('config.yaml')
                user_config = yaml.load(stream, Loader=yaml.FullLoader)
                return user_config

            @st.cache(allow_output_mutation=True)
            def get_token():
                SPOTIPY_CLIENT_ID="53f107ba55324537ab90a576e29d2287"
                SPOTIPY_CLIENT_SECRET="118da4a35c644a5eb8b0c4aefe0f1dd6"
                print("generating token")
                # token = util.prompt_for_user_token(
                #     username=os.environ.get('USERNAME'),
                #     scope='playlist-read-private', 
                #     client_id=os.environ.get('CLIENT_ID'), 
                #     client_secret=os.environ.get('CLIENT_SECRET'), 
                #     redirect_uri=os.environ.get('REDIRECT_URI'))
                # sp = spotipy.Spotify(auth=token)
                sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="53f107ba55324537ab90a576e29d2287", client_secret="118da4a35c644a5eb8b0c4aefe0f1dd6"))

                return sp

            # A function to extract track names and URIs from a playlist
            def get_playlist_info(username, playlist_uri):
                # initialize vars
                offset = 0
                tracks, uris, names, artists = [], [], [], []

                # get playlist id and name from URI
                playlist_id = playlist_uri.split(':')[2]
                playlist_name = sp.user_playlist(username, playlist_id)['name']

                # get all tracks in given playlist (max limit is 100 at a time --> use offset)
                while True:
                    results = sp.user_playlist_tracks(username, playlist_id, offset=offset)
                    tracks += results['items']
                    if results['next'] is not None:
                        offset += 100
                    else:
                        break
                    
                # get track metadata
                for track in tracks:
                    names.append(track['track']['name'])
                    artists.append(track['track']['artists'][0]['name'])
                    uris.append(track['track']['uri'])
                
                return playlist_name, names, artists, uris

            @st.cache(allow_output_mutation=True)
            def get_features_for_playlist(username, uri):
                # initialize_df
                global columns
                temp_df = pd.DataFrame(columns=columns)

                # get all track metadata from given playlist
                playlist_name, names, artists, uris = get_playlist_info(username, uri)
                
                # iterate through each track to get audio features and save data into dataframe
                for name, artist, track_uri in zip(names, artists, uris):
                    
                    # access audio features for given track URI via spotipy 
                    audio_features = sp.audio_features(track_uri)

                    # get relevant audio features
                    feature_subset = [audio_features[0][col] for col in temp_df.columns if col not in ["name", "artist", "track_URI", "playlist"]]

                    # compose a row of the dataframe by flattening the list of audio features
                    row = [name, artist, track_uri, playlist_name, *feature_subset]
                    temp_df.loc[len(temp_df.index)] = row
                return temp_df

            def optimal_number_of_clusters(wcss):
                x1, y1 = 2, wcss[0]
                x2, y2 = 20, wcss[len(wcss)-1]
                distances = []
                for i in range(len(wcss)):
                    x0 = i+2
                    y0 = wcss[i]
                    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
                    denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
                    distances.append(numerator/denominator)
                return distances.index(max(distances)) + 1

            def visualize_data(df, x_axis, y_axis, n_clusters, range_):
                graph = alt.Chart(df.reset_index()).mark_bar().encode(
                    x=alt.X('name', sort='y'),
                    y=alt.Y(str(y_axis)+":Q"),
                    color=alt.Color('cluster', scale=alt.Scale(domain=[i for i in range(n_clusters)], range=range_)),
                    tooltip=['name', 'artist']
                ).interactive()
                st.altair_chart(graph, use_container_width=True)

            def num_components_graph(ax, num_columns, evr):
                ax.plot(range(1, num_columns+1), evr.cumsum(), 'bo-')
                ax.set_title('Explained Variance by Components')
                ax.set(xlabel='Number of Components', ylabel='Cumulative Explained Variance')
                ax.hlines(0.8, xmin=1, xmax=num_columns, linestyles='dashed')
                return ax

            def num_clusters_graph(ax, max_clusters, wcss):
                ax.plot([i for i in range(1, max_clusters)], wcss, 'bo-')
                ax.set_title('Optimal Number of Clusters')
                ax.set(xlabel='Number of Clusters [k]', ylabel='Within Cluster Sum of Squares (WCSS)')
                ax.vlines(KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee, ymin=0, ymax=max(wcss), linestyles='dashed')
                return ax

            @st.cache(allow_output_mutation=True)
            def kmeans(df):
                df_X = df.drop(columns=df.columns[:4])
                print("Standard scaler and PCA")
                scaler = StandardScaler()
                X_std = scaler.fit_transform(df_X) 
                pca = PCA()
                pca.fit(X_std)
                evr = pca.explained_variance_ratio_
                for i, exp_var in enumerate(evr.cumsum()):
                    if exp_var >= 0.8:
                        n_comps = i + 1
                        break
                print("Finding optimal number of components", n_comps)
                pca = PCA(n_components=n_comps)
                pca.fit(X_std)
                scores_pca = pca.transform(X_std)
                wcss = []
                max_clusters = 11
                for i in range(1, max_clusters):
                    kmeans_pca = KMeans(i, init='k-means++', random_state=42)
                    kmeans_pca.fit(scores_pca)
                    wcss.append(kmeans_pca.inertia_)
                n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
                print("Finding optimal number of clusters", n_clusters)
                # fig, (ax1, ax2) = plt.subplots(1, 2)
                # ax1 = num_components_graph(ax1, len(df_X.columns), evr)
                # ax2 = num_clusters_graph(ax2, max_clusters, wcss)
                # fig.tight_layout()
                print("Performing KMeans")
                kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                kmeans_pca.fit(scores_pca)
                df_seg_pca_kmeans = pd.concat([df_X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
                df_seg_pca_kmeans.columns.values[(-1 * n_comps):] = ["Component " + str(i+1) for i in range(n_comps)]
                df_seg_pca_kmeans['Cluster'] = kmeans_pca.labels_
                df['Cluster'] = df_seg_pca_kmeans['Cluster']
                df['Component 1'] = df_seg_pca_kmeans['Component 1']
                df['Component 2'] = df_seg_pca_kmeans['Component 2']
                return df, n_clusters

            @st.cache(allow_output_mutation=True)
            def get_color_range(n_clusters):
                cmap = cm.get_cmap('tab20b')    
                range_ = []
                for i in range(n_clusters):
                    color = 'rgb('
                    mapped = cmap(i/n_clusters)
                    for j in range(3):
                        color += str(int(mapped[j] * 255))
                        if j != 2:
                            color += ", "
                        else:
                            color += ")"
                    range_.append(color)
                return range_

            def visualize_clusters(df, n_clusters, range_):
                graph = alt.Chart(df.reset_index()).mark_point(filled=True, size=60).encode(
                    x=alt.X('Component 2'),
                    y=alt.Y('Component 1'),
                    shape=alt.Shape('playlist', scale=alt.Scale(range=["circle", "diamond", "square", "triangle-down", "triangle-up"])),
                    color=alt.Color('Cluster', scale=alt.Scale(domain=[i for i in range(n_clusters)], range=range_)),
                    tooltip=['name', 'artist']
                ).interactive()
                st.altair_chart(graph, use_container_width=True)

            @st.cache(allow_output_mutation=True)
            def make_normalized_df(df, col_sep):
                print(len(df))
                non_features = df[df.columns[:col_sep]]
                features = df[df.columns[col_sep:]]
                norm = MinMaxScaler().fit_transform(features)
                scaled = pd.DataFrame(norm, index=df.index, columns = df.columns[col_sep:])
                return pd.concat([non_features, scaled], axis=1)

            @st.cache(allow_output_mutation=True)
            def make_radar_chart(norm_df, n_clusters):
                fig = go.Figure()
                cmap = cm.get_cmap('tab20b')
                angles = list(norm_df.columns[5:])
                angles.append(angles[0])

                layoutdict = dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                            ))
                maxes = dict()

                for i in range(n_clusters):
                    subset = norm_df[norm_df['cluster'] == i]
                    data = [np.mean(subset[col]) for col in angles[:-1]]
                    maxes[i] = data.index(max(data))
                    data.append(data[0])
                    fig.add_trace(go.Scatterpolar(
                        r=data,
                        theta=angles,
                        # fill='toself',
                        # fillcolor = 'rgba' + str(cmap(i/n_clusters)),
                        mode='lines',
                        line_color='rgba' + str(cmap(i/n_clusters)),
                        name="Cluster " + str(i)))
                    
                fig.update_layout(
                        polar=layoutdict,
                        showlegend=True
                )
                fig.update_traces()
                return fig, maxes

            @st.cache(allow_output_mutation=True)
            def preview_cluster_playlist(df, cluster):
                df = df[df['cluster'] == cluster]

                # if st.button("Export to playlist"):
                #     result = sp.user_playlist_create(user_config['username'], 'cluster'+str(cluster), public=True, collaborative=False, description='')
                #     playlist_id = result['id']
                #     songs = list(df.loc[df['cluster'] == cluster]['track_URI'])
                #     if len(songs) > 100:
                #         sp.playlist_add_items(playlist_id, songs[:100])
                #         sp.playlist_add_items(playlist_id, songs[100:])
                #     else:
                #         sp.playlist_add_items(playlist_id, songs)
                # else:
                #     pass
                return df

            if __name__ == "__main__":
                # user_config = load_config()
                
                # Initialize Spotify API token
                sp = get_token()
                main()

            

    else:
        st.write("Please select track from the list.")
              



elif selected_album is not None and len(albums) > 0:
    albums_list = albums['albums']['items']
    album_id = None
    album_uri = None    
    album_name = None
    if len(albums_list) > 0:
        for album in albums_list:
            str_temp = album['name'] + " - By - " + album['artists'][0]['name']
            if selected_album == str_temp:
                album_id = album['id']
                album_uri = album['uri']
                album_name = album['name']
    if album_id is not None and album_uri is not None:
        st.write("Collecting all the tracks for the album :" + album_name)
        album_tracks = sp.album_tracks(album_id)
        df_album_tracks = pd.DataFrame(album_tracks['items'])
        # st.dataframe(df_album_tracks)
        df_tracks_min = df_album_tracks.loc[:,
                        ['id', 'name', 'duration_ms', 'explicit', 'preview_url']]
        # st.dataframe(df_tracks_min)
        for idx in df_tracks_min.index:
            with st.container():
                col1, col2, col3, col4 = st.columns((4,4,1,1))
                col11, col12 = st.columns((8,2))
                col1.write(df_tracks_min['id'][idx])
                col2.write(df_tracks_min['name'][idx])
                col3.write(df_tracks_min['duration_ms'][idx])
                col4.write(df_tracks_min['explicit'][idx])   
                if df_tracks_min['preview_url'][idx] is not None:
                    col11.write(df_tracks_min['preview_url'][idx])  
                    with col12:   
                        st.audio(df_tracks_min['preview_url'][idx], format="audio/mp3")              


                               
                        
if selected_artist is not None and len(artists) > 0:
    artists_list = artists['artists']['items']
    artist_id = None
    artist_uri = None
    selected_artist_choice = None
    if len(artists_list) > 0:
        for artist in artists_list:
            if selected_artist == artist['name']:
                artist_id = artist['id']
                artist_uri = artist['uri']
    
    if artist_id is not None:
        artist_choice = ['Albums', 'Top Songs']
        selected_artist_choice = st.sidebar.selectbox('Select artist choice', artist_choice)
                
    if selected_artist_choice is not None:
        if selected_artist_choice == 'Albums':
            artist_uri = 'spotify:artist:' + artist_id
            album_result = sp.artist_albums(artist_uri, album_type='album') 
            all_albums = album_result['items']
            col1, col2, col3 = st.columns((6,4,2))
            for album in all_albums:
                col1.write(album['name'])
                col2.write(album['release_date'])
                col3.write(album['total_tracks'])
        elif selected_artist_choice == 'Top Songs':
            artist_uri = 'spotify:artist:' + artist_id
            top_songs_result = sp.artist_top_tracks(artist_uri)
            for track in top_songs_result['tracks']:
                with st.container():
                    col1, col2, col3, col4 = st.columns((4,4,2,2))
                    col11, col12 = st.columns((10,2))
                    col21, col22 = st.columns((11,1))
                    col31, col32 = st.columns((11,1))
                    col1.write(track['id'])
                    col2.write(track['name'])
                    if track['preview_url'] is not None:
                        #col11.write(track['preview_url'])  
                        with col12:   
                            st.audio(track['preview_url'], format="audio/mp3")  
                    with col4:
                        def similar_songs_requested():
                            token = songrecommendations.get_token(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
                            similar_songs_json = songrecommendations.get_track_recommendations(track['id'], token)
                            recommendation_list = similar_songs_json['tracks']
                            recommendation_list_df = pd.DataFrame(recommendation_list)
                            recommendation_df = recommendation_list_df[['name', 'explicit', 'duration_ms', 'popularity']]
                            with col21:
                                st.dataframe(recommendation_df)
                            with col31:
                                songrecommendations.song_recommendation_vis(recommendation_df)

                        similar_songs_state = st.button('Similar Songs', key=track['id'], on_click=similar_songs_requested)
                    



                        