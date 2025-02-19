import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import folium
from folium import Map, Marker, Popup, IFrame
import dash_html_components as html
import dash_core_components as dcc
import json
import dash_table
from sklearn.cluster import KMeans  # only import once
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go

# Initialisation de l'application Dash
app = dash.Dash(__name__)

# Charger les données
medianes = pd.read_csv('land-registry-house-prices-borough.csv')
age_batiment = pd.read_csv("london_houses.csv")
recycl = pd.read_csv("Household-rcycling-borough.csv")

# Nettoyage des données
medianes["Year"] = medianes["Year"].astype(str).str.extract(r'(\d{4})').astype(int)
medianes["Value"] = medianes["Value"].str.replace(",", "").astype(float)
age_batiment = age_batiment.rename(columns={'Price (£)': 'Price'})

# Moyenne des taux de recyclage par quartier
moyenne_recyclage = recycl.groupby("Area")["Recycling_Rates"].mean().reset_index()
moyenne_recyclage = moyenne_recyclage.sort_values(by="Recycling_Rates", ascending=False)

london_boroughs_coords = {
    "City of London": [51.51279, -0.09184],
    "Barking and Dagenham": [51.5362, 0.1275],
    "Barnet": [51.6255, -0.1514],
    "Bexley": [51.455, 0.1502],
    "Brent": [51.5588, -0.2817],
    "Bromley": [51.4054, 0.0144],
    "Camden": [51.5416, -0.1431],
    "Croydon": [51.3751, -0.0982],
    "Ealing": [51.513, -0.3086],
    "Enfield": [51.6623, -0.118],
    "Greenwich": [51.4892, 0.0648],
    "Hackney": [51.553, -0.06],
    "Hammersmith and Fulham": [51.4927, -0.2213],
    "Haringey": [51.5908, -0.1097],
    "Harrow": [51.5788, -0.3337],
    "Havering": [51.5761, 0.1837],
    "Hillingdon": [51.5352, -0.4482],
    "Hounslow": [51.4746, -0.3702],
    "Islington": [51.5441, -0.1026],
    "Kensington and Chelsea": [51.4995, -0.1936],
    "Kingston upon Thames": [51.4123, -0.3007],
    "Lambeth": [51.488, -0.1154],
    "Lewisham": [51.4415, -0.0117],
    "Merton": [51.4098, -0.2102],
    "Newham": [51.5255, 0.0352],
    "Redbridge": [51.5886, 0.0822],
    "Richmond upon Thames": [51.4452, -0.3294],
    "Southwark": [51.4979, -0.0758],
    "Sutton": [51.359, -0.191],
    "Tower Hamlets": [51.5155, -0.0371],
    "Waltham Forest": [51.5884, -0.0112],
    "Wandsworth": [51.4576, -0.191],
    "Westminster": [51.4975, -0.1357],
}

def generate_recycling_map():
    # Calcul du taux moyen de recyclage par quartier
    avg_recycling = recycl.groupby("Area")["Recycling_Rates"].mean().to_dict()

    # Création de la carte centrée sur Londres
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    # Fonction pour déterminer la couleur en fonction du taux de recyclage
    def get_recycling_color(rate):
        if rate <= 15:
            return "red"  # Faible
        elif rate <= 30:
            return "orange"  # Moyenne
        elif rate <= 45:
            return "yellow"  # Élevée
        else:
            return "green"  # Très élevée

    # Ajout de marqueurs pour chaque quartier
    for borough, coords in london_boroughs_coords.items():
        if borough in avg_recycling:
            mean_rate = avg_recycling[borough]
            
            # Création du graphique d’évolution avec Plotly
            borough_data = recycl[recycl["Area"] == borough]
            fig = px.line(
                borough_data,
                x="Year",
                y="Recycling_Rates",
                title=f"Évolution du taux de recyclage à {borough}",
                labels={"Year": "Année", "Recycling_Rates": "Taux de recyclage (%)"},
            )

            # Conversion du graphique en HTML
            iframe = IFrame(fig.to_html(full_html=False, include_plotlyjs='cdn'), width=500, height=300)
            popup = Popup(iframe, max_width=500)

            # Ajout d’un cercle sur la carte avec la couleur selon le taux de recyclage
            folium.CircleMarker(
                location=coords,
                radius=10,
                color="blue",
                fill=True,
                fill_color=get_recycling_color(mean_rate),  # Couleur selon le taux
                fill_opacity=0.6,
                popup=popup,
            ).add_to(m)

    # Ajout d’une légende manuelle
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; z-index:9999; font-size:14px;
                padding: 10px; border-radius: 5px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        <b>Légende :</b><br>
        <i class="fa fa-circle" style="color:red"></i> Faible (≤15%)<br>
        <i class="fa fa-circle" style="color:orange"></i> Moyenne (≤30%)<br>
        <i class="fa fa-circle" style="color:yellow"></i> Élevée (≤45%)<br>
        <i class="fa fa-circle" style="color:green"></i> Très élevée (>45%)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Sauvegarde de la carte en HTML
    map_path = "carte_recyclage.html"
    m.save(map_path)
    
    return map_path

Prix2 = pd.read_csv("London.csv")
# Nettoyer les noms de colonnes pour éviter les erreurs avec LightGBM
Prix2.columns = Prix2.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

#pour la section 3

def generate_cluster_repartition_and_intervals(kmeans_model, cluster_column_name):
    cluster_centroids = kmeans_model.cluster_centers_
    cluster_intervals = sorted(cluster_centroids.flatten())  # Tri des centroids pour obtenir des intervalles
    
    # Ajout des bornes minimales et maximales pour créer les intervalles de prix
    min_price, max_price = Prix2['Price'].min(), Prix2['Price'].max()
    cluster_intervals = [min_price] + cluster_intervals + [max_price]
    
    # Répartition des données dans chaque cluster
    cluster_repartition = Prix2[cluster_column_name].value_counts().sort_index()

    # Créer une liste des intervalles de prix
    price_intervals = [f"{cluster_intervals[i]} - {cluster_intervals[i + 1]}" for i in range(len(cluster_intervals) - 1)]

    # DataFrame pour la répartition des clusters et des intervalles de prix
    cluster_repartition_df = pd.DataFrame({
        "Cluster": cluster_repartition.index,
        "Count": cluster_repartition.values,
        "Price Interval": price_intervals[:len(cluster_repartition)]  # Ajuster la taille des intervalles
    })
    
    return cluster_repartition_df

X = Prix2[['Price']] 

# Create the KMeans model with 5 clusters
kmeans_5 = KMeans(n_clusters=5, random_state=42)
Prix2['Price Category (5 Clusters)'] = kmeans_5.fit_predict(X)
# For kmeans_4 with 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42)
Prix2['Price Category (4 Clusters)'] = kmeans_4.fit_predict(X)

# For kmeans_6 with 6 clusters
kmeans_6 = KMeans(n_clusters=6, random_state=42)
Prix2['Price Category (6 Clusters)'] = kmeans_6.fit_predict(X)

# For kmeans_7 with 7 clusters
kmeans_7 = KMeans(n_clusters=7, random_state=42)
Prix2['Price Category (7 Clusters)'] = kmeans_7.fit_predict(X)

# For kmeans_8 with 8 clusters
kmeans_8 = KMeans(n_clusters=8, random_state=42)
Prix2['Price Category (8 Clusters)'] = kmeans_8.fit_predict(X)



# 1. Générer les tableaux pour chaque nombre de clusters
cluster_4_df = generate_cluster_repartition_and_intervals(kmeans_4, 'Price Category (4 Clusters)')
cluster_5_df = generate_cluster_repartition_and_intervals(kmeans_5, 'Price Category (5 Clusters)')
cluster_6_df = generate_cluster_repartition_and_intervals(kmeans_6, 'Price Category (6 Clusters)')
cluster_7_df = generate_cluster_repartition_and_intervals(kmeans_7, 'Price Category (7 Clusters)')
cluster_8_df = generate_cluster_repartition_and_intervals(kmeans_8, 'Price Category (8 Clusters)')


cluster_data = {
    "4 Clusters": cluster_4_df,
    "5 Clusters": cluster_5_df,
    "6 Clusters": cluster_6_df,
    "7 Clusters": cluster_7_df,
    "8 Clusters": cluster_8_df
}

# Répartition des données d'apprentissage pour chaque catégorie de prix (clusters)
y_4_clusters = Prix2['Price Category (4 Clusters)']
y_5_clusters = Prix2['Price Category (5 Clusters)']
y_6_clusters = Prix2['Price Category (6 Clusters)']
y_7_clusters = Prix2['Price Category (7 Clusters)']
y_8_clusters = Prix2['Price Category (8 Clusters)']


# Sélectionner les colonnes que vous souhaitez inclure dans X
features = ['House_Type', 'Area_in_sq_ft', 'No__of_Bedrooms', 
            'No__of_Bathrooms', 'No__of_Receptions', 'Location']

X = Prix2[features]

# Encoder les variables catégorielles (exemple pour 'House_Type' et 'Location')
label_encoder = LabelEncoder()

# Applique l'encodage sur les variables catégorielles
X['House_Type'] = label_encoder.fit_transform(X['House_Type'])
X['Location'] = label_encoder.fit_transform(X['Location'])

# Pour 'Property_Name', vous pouvez faire de même si vous voulez l'utiliser aussi.
# Notez que l'encodage de 'Property_Name' peut ne pas être utile selon la nature de cette variable,
# mais si vous voulez l'utiliser, vous pouvez aussi appliquer un encodage similaire.

# Séparation des données en train et test pour les différentes catégories de prix (clusters)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X, y_4_clusters, test_size=0.2, random_state=42)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X, y_5_clusters, test_size=0.2, random_state=42)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X, y_6_clusters, test_size=0.2, random_state=42)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(X, y_7_clusters, test_size=0.2, random_state=42)
X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(X, y_8_clusters, test_size=0.2, random_state=42)

# Modèles de classification
models = {
    "XGBoost": XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.1, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=1000, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
}

# Stockage des performances
results = []

for name, model in models.items():
    for decoupage, X_train, X_test, y_train, y_test in [
        ("4 Clusters", X_train_4, X_test_4, y_train_4, y_test_4),
        ("5 Clusters", X_train_5, X_test_5, y_train_5, y_test_5),
        ("6 Clusters", X_train_6, X_test_6, y_train_6, y_test_6),
        ("7 Clusters", X_train_7, X_test_7, y_train_7, y_test_7),
        ("8 Clusters", X_train_8, X_test_8, y_train_8, y_test_8)
    ]:
        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation
        acc = accuracy_score(y_test, y_pred)
        results.append((name, decoupage, acc))

# Création d'un DataFrame pour stocker les résultats
df_results = pd.DataFrame(results, columns=['Modèle', 'Découpage', 'Accuracy'])



# Entraîner le modèle RandomForest sur les données des 5 clusters
rf_model_5 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_5.fit(X_train_5, y_train_5)

# Extraire l'importance des variables
feature_importances = rf_model_5.feature_importances_

# Créer un DataFrame pour afficher les résultats
importance_df = pd.DataFrame({
    'Feature': X_train_5.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Créer un graphique de l'importance des variables
importance_fig = px.bar(importance_df, x='Feature', y='Importance', 
                        title="Importance des variables pour la classification avec Random Forest (5 Clusters)",
                        labels={'Importance': 'Importance', 'Feature': 'Variable'},
                        color='Importance', color_continuous_scale='Viridis')


# Charger les données
prix_moyen = pd.read_csv("moyennes_prix.csv", sep=";")
prix_moyen = prix_moyen.drop(columns=prix_moyen.columns[2])  # Garde toutes les colonnes sauf la première
# Extraire les années uniques en supprimant le texte "Year ending XXX"
prix_moyen_annees=prix_moyen.drop(columns=prix_moyen.columns[0])
prix_moyen_annees=prix_moyen_annees.drop(columns=prix_moyen_annees.columns[0])
# Extraire les années uniques en supprimant "Year ending XXX"
prix_moyen_annees.columns = prix_moyen_annees.columns.str.extract(r'(\d{4})')[0]
# Remplacer les espaces insécables (code Unicode) par des espaces simples ou rien
prix_moyen_annees = prix_moyen_annees.replace(r'\s+', '', regex=True)

# Convertir les valeurs en numérique, forcer la conversion en NaN si nécessaire
prix_moyen_annees = prix_moyen_annees.apply(pd.to_numeric, errors='coerce')

# Convertir les valeurs en numérique pour éviter les erreurs de type
prix_moyen_annees = prix_moyen_annees.apply(pd.to_numeric, errors='coerce')

# Vérifier le nombre de colonnes
n_colonnes = prix_moyen_annees.shape[1]

data_array = prix_moyen_annees.to_numpy()
# Calculer la moyenne tous les 3 colonnes
moyennes = np.nanmean(data_array.reshape(prix_moyen_annees.shape[0], -1, 4), axis=2)

# Générer les noms des nouvelles colonnes (en supposant que les années commencent à 1996)
annees = list(range(1996, 1996 + moyennes.shape[1]))  
new_columns = [f"moyenne_{annee}" for annee in annees]

# Créer le DataFrame avec les moyennes
moyennes_annuelles = pd.DataFrame(moyennes, columns=new_columns)

# Ajouter les colonnes "Code" et "Area" du DataFrame original
moyennes_annuelles.insert(0, "Code", prix_moyen["Code"])
moyennes_annuelles.insert(1, "Area", prix_moyen["Area"])

recyclage=pd.read_csv("Household-rcycling-borough.csv")
dispo=pd.read_csv("Dclg-affordable-housing-borough.csv")
def transform_year(year):
    # Si l'année suit le format "YYYY-YY" (comme "2003-04")
    if isinstance(year, str) and '-' in year:
        return int(year.split("-")[0])
    # Si l'année suit le format "YYYY/YY" (comme "2003/04")
    elif isinstance(year, str) and '/' in year:
        return int(year.split("/")[0])
    else:
        return year  # retourne la valeur d'origine en cas de format inconnu

# Transformation des années pour dispo uniquement
dispo['Year'] = dispo['Year'].apply(transform_year)

# Transformation des années pour recyclage
recyclage['Year'] = recyclage['Year'].apply(transform_year)

# Pivoté les données du recyclage pour que les années deviennent des colonnes
recyclage_pivot = recyclage.pivot_table(index=['Code', 'Area'], columns='Year', values='Recycling_Rates', aggfunc='mean')

# Supprimer les virgules dans 'Affordable Housing Supply'
dispo['Affordable Housing Supply'] = dispo['Affordable Housing Supply'].str.replace(',', '')

# Convertir la colonne en numérique
dispo['Affordable Housing Supply'] = pd.to_numeric(dispo['Affordable Housing Supply'], errors='coerce')

# Pivot des données : la colonne Year devient les colonnes et les valeurs sont dans 'Affordable Housing Supply'
dispo_pivoted = dispo.pivot_table(index=['Code', 'Area'], columns='Year', values='Affordable Housing Supply', aggfunc='mean')
# Renommer les colonnes pour qu'elles commencent par 'dispo_'
dispo_pivoted.columns = [f"dispo_{annee}" for annee in dispo_pivoted.columns]
recyclage_pivot.columns = [f"recly_{annee}" for annee in recyclage_pivot.columns]
df = pd.merge(moyennes_annuelles,recyclage_pivot, on=['Code', 'Area'], how='left')
df2=pd.merge(df,dispo_pivoted, on=['Code', 'Area'], how='left')
df2 = df2.dropna()

# Associer les coordonnées aux boroughs de Londres
df2["Latitude"] = df2["Area"].map(lambda x: london_boroughs_coords.get(x, [None, None])[0])
df2["Longitude"] = df2["Area"].map(lambda x: london_boroughs_coords.get(x, [None, None])[1])

df2 = df2.dropna(subset=["Latitude", "Longitude"])

# Liste des colonnes à exclure (exemple : identifiants, noms, etc.)
colonnes_a_exclure = ["Code", "Area", "Latitude", "Longitude"]  

# Sélectionner toutes les autres colonnes
features = df2.columns.difference(colonnes_a_exclure)  

# Créer le DataFrame pour le clustering
df_cluster = df2[features].dropna()

# Convertir toutes les colonnes en numérique, en ignorant les erreurs (coercition en NaN si non convertible)
df_cluster = df_cluster.apply(pd.to_numeric, errors='coerce')

# Normalisation des données
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# Appliquer K-Means avec le bon nombre de clusters
kmeans = KMeans(n_clusters=4, random_state=42)  # Remplace 4 par le bon k trouvé
df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

# Ajouter les clusters au DataFrame principal
df2['Cluster'] = df_cluster['Cluster']

fig_clusters = px.scatter_mapbox(df2, lat="Latitude", lon="Longitude", color="Cluster",
                        hover_name="Area", zoom=10, mapbox_style="carto-positron")


cluster_means = df2.groupby("Cluster").mean()

# Sélection des colonnes des prix
prix_cols = [col for col in df2.columns if "moyenne_" in col]

# Moyenne des prix par année et par cluster
prix_cluster = df2.groupby("Cluster")[prix_cols].mean().T


fig_prix_cluster = go.Figure()

for cluster in prix_cluster.columns:
    fig_prix_cluster.add_trace(go.Scatter(
        x=prix_cluster.index.str.replace("moyenne_", ""), 
        y=prix_cluster[cluster],
        mode='lines',
        name=f'Cluster {cluster}'
    ))

fig_prix_cluster.update_layout(
    title="Évolution des prix moyens par cluster",
    xaxis_title="Année",
    yaxis_title="Prix moyen",
    template="plotly_white"
)


# Sélection des colonnes de disponibilité
dispo_cols = [col for col in df2.columns if "dispo_" in col]

# Moyenne des disponibilités par année et par cluster
dispo_cluster = df2.groupby("Cluster")[dispo_cols].mean().T

# Créer une figure Plotly pour les disponibilités
fig_dispo_cluster = go.Figure()

for cluster in dispo_cluster.columns:
    fig_dispo_cluster.add_trace(go.Scatter(
        x=dispo_cluster.index.str.replace("dispo_", ""), 
        y=dispo_cluster[cluster],
        mode='lines',
        name=f'Cluster {cluster}'
    ))

fig_dispo_cluster.update_layout(
    title="Évolution de la disponibilité des logements par cluster",
    xaxis_title="Année",
    yaxis_title="Disponibilité moyenne",
    template="plotly_white"
)


# Sélection des colonnes de recyclage
recy_cols = [col for col in df2.columns if "recly_" in col]

# Moyenne des taux de recyclage par année et par cluster
recy_cluster = df2.groupby("Cluster")[recy_cols].mean().T

# Créer une figure Plotly pour le taux de recyclage
fig_recy_cluster = go.Figure()

for cluster in recy_cluster.columns:
    fig_recy_cluster.add_trace(go.Scatter(
        x=recy_cluster.index.str.replace("recly_", ""), 
        y=recy_cluster[cluster],
        mode='lines',
        name=f'Cluster {cluster}'
    ))

fig_recy_cluster.update_layout(
    title="Évolution du taux de recyclage par cluster",
    xaxis_title="Année",
    yaxis_title="Taux de recyclage moyen",
    template="plotly_white"
)



# Layout de l'application Dash
app.layout = html.Div([
    html.H1("Analyse des logements à Londres"),
        html.P("Ce projet vise à analyser le marché du logement à Londres en exploitant diverses sources de données. L'objectif est de comprendre l'évolution des prix, des ventes et des caractéristiques des logements dans les différents quartiers (33) de la ville et d'identifier des comportements similaires entre quartiers."),
    dcc.Markdown('''
    L'analyse s'est faite en plusieurs étapes : 

    - **Statistiques descriptives** : étude de l'évolution des prix par quartier, du nombre de ventes et des potentiels facteurs ayant une influence sur le prix des logements.
    - **Clustering** : L'objectif ici est de regrouper des quartiers ensemble autour de certaines caractéristiques.
    - **Classification** : identification des caractéristiques permettant de prédire l’appartenance d’un quartier à une tranche de prix donnée.

    Les données utilisées proviennent de sources officielles et ouvertes, notamment :
    - **Housing-sales-borough.csv** : ventes immobilières par quartier (1995-2014).
    - **land-registry-house-prices-borough.csv** : prix médians des logements (1995-2017).
    - **Dclg-affordable-housing-borough.csv** : logements abordables construits par année et quartier.
    - **Household-recycling-borough.csv** : taux de recyclage par quartier (2003-2023).
    - **Tenure-households-borough.csv & Tenure-population-borough.csv** : répartition des logements et population par type d’occupation (2008-2018).
    - **London.csv & London_houses.csv** : caractéristiques détaillées des logements londoniens (prix, superficie, nombre de chambres, etc.).

    Cette analyse a pour objectif global de comprendre les dynamiques du marché des logements londoniens.
    '''),

    # Section 1 : Statistiques descriptives
    html.H2("1. Statistiques descriptives"),

    # 1.1 Évolution des prix des logements
    html.H3("1.1 Évolution des prix des logements"),
    html.P([
    "Regardons l'évolution du prix moyen des logements à Londres d'année en année depuis 1995. Voici le jeu de données utilisé : ",
    html.A("land-registry-house-prices-borough.csv", 
           href="https://data.london.gov.uk/download/average-house-prices/f01b1cc7-6daa-4256-bd6c-94d8c83ee000/land-registry-house-prices-borough.xls", 
           target="_blank")
]),

    dcc.Graph(id='prix-maisons-graph'),  # Graphique 1

    html.P("Sélectionnez un quartier pour voir l'évolution des prix."),
    dcc.Dropdown(
        id='dropdown-quartier',
        options=[{'label': area, 'value': area} for area in medianes["Area"].unique()],
        value='London',  # Quartier par défaut
        style={'width': '50%'}
    ),
    dcc.Graph(id='graph-quartier'),  # Graphique 2

    html.P("On observe une augmentation (attendue) des prix des logements. On peut noter que le prix moyen augmente plus rapidement que le prix médian suggérant que les prix des propriétés les plus chères augmentent plus rapidement que ceux des propriétés moins chères."),
    dcc.Graph(id='boxplot-prix'),# Graphique 3
    html.P("Ces boxplot permettent de voir que la distribution des prix est inégale selon les quartiers et donc que le quartier d'appartenance a surement une influence sur le prix. Un clustering des quartiers peut donc être intéressant."),
    html.P("Regardons maintenant si certaines variables ont une influence sur le prix."),
    
    # 1.2 Distribution des prix en fonction du statut de rénovation
    html.H3("1.2 Distribution des prix en fonction du statut de rénovation"),
    html.P([
    "Les prochains graphiques ont été obtenus grâce au tableau de données : donc voici le lien : ",
    html.A("Household-rcycling-borough.csv",
           href="https://data.london.gov.uk/download/household-waste-recycling-rates-borough/15ddc38a-0a37-4f69-98b5-e69e549b39d3/Household-rcycling-borough.csv",
           target="_blank")
    ]),
    html.P("Ce graphique montre la distribution des prix selon que le logement a été rénové ou non."),
    dcc.Graph(id='boxplot-renovation'),  # Graphique 4
    html.P("On peut observer ici un résultat plutôt attendu :la médiane des prix des logements rénovés ou neufs est plus élevée que ceux qui sont vieux"),

    # 1.3 Taux de recyclage des logements de Londres
    html.H3("1.3 Taux de recyclage des logements de Londres"),
    html.P("Ayant accès à des données de recyclage nous avons souhaité regarder si il y avait une tendance géographique de recyclage afin de savoir si c'était une donnée qui pourrait être pertinente pour le clustering de quartiers. "),
    dcc.Graph(id='graph-recyclage'),  # Graphique 5
    html.P("On remarque que le taux de recyclage varie d'un quartier à l'autre. Il serait intéressant de le visualiser sur une carte afin de détecter d'éventuelles tendances géographiques."),

    html.Iframe(id="recycling-map",srcDoc=open(generate_recycling_map(), "r", encoding="utf-8").read(),width="100%",height="600px",style={"border": "none"}),
    html.P("Les habitants des quartiers extérieurs de Londres trient davantage leurs déchets que les individus habitant dans le centre. Le taux de recyclage pourra être pris en compte dans le clustering de quartiers."),
    
    # Section 2 : Clustering
    html.H2("2. Clustering"),
    html.P("L'objectif ici est de trouver des clusters de quartiers, à l'aide du jeu de données que l'on peut charger sur le lien suivant : mettre le lien"),
    html.P("Ce jeu de données comprend les ventes, les prix médians et moyens des logements 3 fois par an de 1995 à 2022."),
    html.P("À ce jeu de données on a ajouté d'autres jeux de données sur les taux de recyclage déjà utilisé précédemment ainsi qu'un autre jeu de données sur les logements disponibles (lien)."),
    html.P("Des modifications ont été faites sur le nom des colonnes et les datasets ont été fusionnés. Ensuite les données ont été normalisées. Le clustering a été appliqué à l'aide de la méthode des k-means avec le k optimal trouvé grâce à la méthode du coude."),
    html.P("La méthode des k-means consiste à choisir k centroïdes initiaux, puis à assigner à chaque point restant le centroïde le plus proche. Puis le centroïde est recalculé. Une fois que l'algorithme converge on s'arrête."),
    html.P("Ici 3 clusters ont été établis. Regardons sur la carte si ils sont répartis d'une certaine manière."),
    dcc.Graph(id="map", figure=fig_clusters),
    html.P("On remarque que les quartiers du cluster 3 sont concentrés au centre de Londres, les quartiers de cluster 2 et 1 sont autour du centre et ceux du cluster 0 sont aux extrémités. Regardons les tendances des clusters selon les prix, le nombre de logements disponibles et le taux de recyclage."),
    dcc.Graph(id="prix-cluster", figure=fig_prix_cluster),
    html.P("Concernant les prix, c'est le cluster 3 qui se distingue des autres, les quartiers du centre ont des logements plus chers que les autres."),
    dcc.Graph(id="dispo-cluster", figure=fig_dispo_cluster), 
    html.P("Concernant le nombre de logements disponibles le cluster 1 a une moyenne légèrement plus élevée que les autres. On ne peut pas tirer de conclusion pertinente concernant la répartition spatiale des logements disponibles."),
    dcc.Graph(id="recy-cluster", figure=fig_recy_cluster),
    html.P("Enfin concernant le taux de recyclage, les quartiers du cluster 0 sont ceux qui recyclent le plus, tandis que ceux du cluster 3 sont ceux qui recyclent le moins, ce qui vient confirmer le graphe du taux de recyclage vu précédemment."),
    html.P("Ainsi, ce clustering de quartiers, permet de se rendre compte qu'il y a des dynamiques différentes selon que l'on se trouve dans un quartier du centre ou excentré notamment en terme de prix et de recyclage."),
    
    #section 3 : classification 
    html.H2("3. Classification d'appartements par tranche de prix"),
    html.P("L'objectif ici est de de comparer différentes méthodes de classification sur le prix des appartements londoniens."),
    html.P("Le jeu de données utilisé est london.csv, ce jeu recense 3480 appartements, et comprend donc 3480 lignes, qui une fois trié (séléction des quartiers de la métropole londonienne) comprend 2972 lignes. Les variables sont : 'House Type', 'Area in sq ft', 'No. of Bedrooms', 'No. of Bathrooms','No. of Receptions', 'Location', 'City/County'. Dans un premier temps nous allons faire un clustering de ces logements afin de créer des classes, puis grâce aux variables nous pourrons faire de la classification à l'aide de différentes méthodes et déterminer celle qui est la plus efficace.",style={'text-align': 'justify'}),
    
    html.H3("3.1 Clustering à l'aide de la méthode des k-means"),
    html.P("Le prix étant une variable continue, nous avons dans un premier temps effectué un clustering à l'aide de la méthode des k-means sur les prix de vente des appartements afin de garder entre 4 et 8 catégories de prix."),
    html.P("Ces tableaux montrent la répartition des données pour les différents clusters des différents clustering."),
    html.H3("Sélectionnez un clustering"),
    
    dcc.Dropdown(
        id="cluster-dropdown",
        options=[{"label": key, "value": key} for key in cluster_data.keys()],
        value="4 Clusters",  # Valeur par défaut
        clearable=False
    ),

    dash_table.DataTable(
        id="cluster-table",
        style_table={'height': '300px', 'overflowY': 'auto'}),
    html.P("En regardant les fréquences de chaque classe pour les différents clustering on peut remarquer que certaines classes sont très peu représentées. Cela induit souvent une mauvaise prédiction de classe en question et provoque donc une diminution de la perte de précisions de prédiction. Idéalement, il faudrait un recensement complet de tous les appartements de Londres, ce qui permettrait d'avoir des classes plus complètes, et d'avoir une meilleure précisiond de prédiction. Il faut maintenant tester différentes méthodes de classification avec tous les clustering sachant que plus on a de catégories meilleure sera la précision du prix puisque la fourchette de prix sera plus petite. Cependant, augmenter le nombre de clusters risque de faire augmenter le nombre d'erreurs et donc la précision de prédiction. Vérifions cette hypothèse.",style={'text-align': 'justify'}),
    html.H3("3.2 Comparaison de différentes méthodes de classification"),
    html.P("Afin de comparer la précision des méthodes, l'ensemble des données a été séparé en ensemble d'apprentissage/ test (80 % des données/ 20 % des données)."),
    html.P("Les modèles sont entrainés sur l'ensemble d'apprentissge et testés sur l'ensemble de test."),
    html.P("La précision (Accuracy) est le quotient entre le nombre de réponses correctes et le nombre total de prédictions."),
    html.P("Les différentes méthodes qui ont été comparées sont : l'arbre décision, la forêt aléatoire, l'algorithme light gbm et l'algorithme xgboost."),
    html.P("Pour plus d'informations sur ces méthodes :"),
    html.A("Abre de décision", href = "https://www.ibm.com/fr-fr/think/topics/decision-trees"),
    html.A("Forêt aléatoire", href = "https://datascientest.com/random-forest-definition"),
    html.A("Xgboost", href = "https://xgboost.readthedocs.io/en/stable/"),
    html.A("LightGbm", href = "ttps://towardsdatascience.com/a-quick-guide-to-lightgbm-library-ef5385db8d10/"),
    
    dcc.Graph(id="classification-results-graph"),  # Graph for classification accuracy
    
    html.P("Ce graphique montre la précision des différentes méthodes de classifications pour chaque clustering"),
    html.P("La méthode random forest semble être la plus performante. LightGBM et XGBoost auraient sûrement été plus performants si nous avions eu plus de données."),
    html.H3("3.3 Variables les plus importantes pour la classification par tranche de prix pour random forest"),
    
    html.P("Ce graphique montre l'importance des variables pour dans la classification avec forêts aléatoires et 5 clusters"),
     # Ajouter le graphique de l'importance des variables pour les 5 clusters
    dcc.Graph(
        id="importance-variables-rf-5-clusters",
        figure=importance_fig  # La figure du graphique de l'importance des variables
    ),
    html.P("Les deux variables les plus déterminantes du prix des appartements londoniens sont la surface (constat évident) et la localisation (le quartier)."),
    html.H2("4. Conclusion"),
    html.P("Cette analyse du marché immobilier londonien a permis d’explorer plusieurs aspects du marché des logements, en mettant en évidence l’évolution des prix d’année en année, des comportements similaires entre quartiers et les facteurs influençant les prix des logements.",style={'text-align': 'justify'}),
    html.P(" Dans un premier temps, l’analyse descriptive a confirmé une augmentation générale des prix à Londres depuis 1995, avec une différence entre les quartiers. L’impact de certaines variable a été osbervé, telles que la rénovation des logements et le taux de recyclage, variables qui semblent avoir des tendances géographiques.",style={'text-align': 'justify'}),
    html.P(" Ensuite, le clustering des quartiers a permis d’identifier des regroupements, dont les quartiers partagent des caractéristiques communes. Cette segmentation a montré des tendances géographiques intéressantes. Enfin, la classification des appartements par tranche de prix a permis d’évaluer différentes méthodes d’apprentissage supervisé. Parmi les algorithmes testés (arbres de décision, forêts aléatoires, LightGBM et XGBoost), la forêt aléatoire a obtenu la meilleure précision (de peu). Les performances de LightGBM et XGBoost sont proches de celles de Random Forest et auraient été sûrement meilleures avec beaucoup plus de données. L’analyse de l’importance des variables a également confirmé que la surface et la localisation étaient les principaux déterminants du prix des appartements londoniens.",style={'text-align': 'justify'}),
    html.P("Cette étude pourrait être approfondie d’une part en ayant accès aux données de plus de logements, pour améliorer et généraliser les modèles prédictifs, notamment LightBGM et XGBoost qui devraient être plus performants que Random Forest. Des méthodes de régression ont aussi été essayées mais n’ont pas pu être abouties. Si les méthodes sont performantes elles pourraient permettre aux acheteurs ou aux vendeurs de déterminer le prix d’un bien de manière fiable.",style={'text-align': 'justify'})

])


# Callback pour afficher l'évolution des prix de Londres
@app.callback(
    Output('prix-maisons-graph', 'figure'),
    Input('prix-maisons-graph', 'id')
)
def update_london_graph(_):
    df_borough = medianes[medianes["Area"] == "London"]
    fig = px.line(df_borough, x="Year", y="Value", color="Measure",
                  title="Évolution des prix des maisons à Londres",
                  labels={"Value": "Prix (£)", "Year": "Année"})
    return fig

# Callback pour mettre à jour le graphique par quartier
@app.callback(
    Output('graph-quartier', 'figure'),
    [Input('dropdown-quartier', 'value')]
)
def update_graph(quartier):
    df_quartier = medianes[medianes["Area"] == quartier]
    fig = px.line(df_quartier, x="Year", y="Value", color="Measure",
                  title=f"Évolution des prix à {quartier}",
                  labels={"Value": "Prix (£)", "Year": "Année"})
    return fig

# Callback pour afficher le graphique des taux de recyclage
@app.callback(
    Output('graph-recyclage', 'figure'),
    Input('graph-recyclage', 'id')
)
def update_recycling_graph(_):
    fig = px.bar(moyenne_recyclage, x="Recycling_Rates", y="Area",
                 title="Moyenne des taux de recyclage par quartier à Londres",
                 labels={"Recycling_Rates": "Taux moyen de recyclage (%)", "Area": "Quartier"},
                 orientation="h",  # Barres horizontales
                 color="Recycling_Rates", color_continuous_scale="viridis")

    fig.update_layout(yaxis=dict(categoryorder='total ascending'))  # Trier de haut en bas
    return fig

# Callback pour afficher le boxplot des prix par quartier
@app.callback(
    Output('boxplot-prix', 'figure'),
    Input('boxplot-prix', 'id')
)
def update_boxplot(_):
    fig = px.box(age_batiment, x="Neighborhood", y="Price",
                 title="Distribution des prix par quartier",
                 labels={"Price": "Prix (£)", "Neighborhood": "Quartier"})
    
    fig.update_layout(xaxis_tickangle=-90)
    return fig

# Callback pour afficher le boxplot des prix par statut de rénovation
@app.callback(
    Output('boxplot-renovation', 'figure'),
    Input('boxplot-renovation', 'id')
)
def update_renovation_boxplot(_):
    fig = px.box(age_batiment, x="Building Status", y="Price",
                 title="Distribution des prix en fonction du statut de rénovation",
                 labels={"Price": "Prix (£)", "Building Status": "Statut de rénovation"})
    
    return fig


@app.callback(
    Output('classification-results-graph', 'figure'),
    Input('classification-results-graph', 'id')
)
def update_classification_results(_):
    # Create a bar plot using Plotly
    fig = px.bar(df_results, 
                 x="Découpage", 
                 y="Accuracy", 
                 color="Modèle", 
                 title="Comparaison des méthodes de classification",
                 labels={"Accuracy": "Précision", "Découpage": "Clustering"},
                 barmode="group")

    # Return the figure to be displayed
    return fig
    
# Callback pour mettre à jour le tableau en fonction de la sélection
@app.callback(
    Output("cluster-table", "columns"),
    Output("cluster-table", "data"),
    Input("cluster-dropdown", "value")
)
def update_table(selected_cluster):
    df = cluster_data[selected_cluster]
    return [{"name": col, "id": col} for col in df.columns], df.to_dict('records')



if __name__ == "__main__":
    # Exporter l'application Dash en HTML
    with open("index.html", "w") as f:
        f.write(app.index_string)  # Écrire la chaîne générée dans le fichier

    # Lancer le serveur Dash
    app.run_server(debug=True)