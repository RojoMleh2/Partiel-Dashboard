import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---- Chargement des données ----
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv", sep=";", encoding="utf-8", engine="python", on_bad_lines="skip")
    return df

df = load_data()

# ---- Sidebar : Filtres ----
st.sidebar.header("Filtres")
age_range = st.sidebar.slider("Tranche d'âge", int(df["Age"].min()), int(df["Age"].max()), (25, 60))
anciennete_range = st.sidebar.slider("Ancienneté", int(df["Ancienneté"].min()), int(df["Ancienneté"].max()), (0, 10))
pays_selection = st.sidebar.multiselect("Pays", df["Localisation"].unique(), default=df["Localisation"].unique())
genre_selection = st.sidebar.radio("Genre", ["Tous", "Homme", "Femme"])

# Filtrage des données
df_filtered = df[
    (df["Age"].between(*age_range)) &
    (df["Ancienneté"].between(*anciennete_range)) &
    (df["Localisation"].isin(pays_selection))
]
if genre_selection != "Tous":
    df_filtered = df_filtered[df_filtered["Genre"] == genre_selection.lower()]

# ---- Page d'accueil ----
st.title("Dashboard Client - Analyse de l'Attrition")

col1, col2 = st.columns(2)
col1.metric("Taux d'attrition global", f"{df['Attrition'].mean() * 100:.2f} %")
col2.metric("Âge moyen des clients partis", f"{df[df['Attrition'] == 1]['Age'].mean():.1f} ans")

# Carte interactive des clients ayant quitté
st.subheader("Répartition géographique des clients en attrition")
df_map = df[df["Attrition"] == 1].groupby("Localisation").size().reset_index(name="Nombre")
fig_map = px.choropleth(df_map, locations="Localisation", locationmode="country names", color="Nombre", 
                         title="Clients ayant quitté la banque")
st.plotly_chart(fig_map)

# ---- Onglets pour analyses détaillées ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Démographie", "Engagement", "Finances", "Relation Client", "Prédictions"])

with tab1:  # Analyse Démographique
    st.header("Analyse Démographique")
    fig_age = px.histogram(df_filtered, x="Age", color="Attrition", barmode="overlay")
    st.plotly_chart(fig_age)
    
    fig_genre = px.bar(df_filtered.groupby("Genre")["Attrition"].mean().reset_index(),
                       x="Genre", y="Attrition", title="Attrition selon le Genre")
    st.plotly_chart(fig_genre)

    fig_pays = px.bar(df_filtered.groupby("Localisation")["Attrition"].mean().reset_index(),
                      x="Localisation", y="Attrition", title="Attrition par Pays")
    st.plotly_chart(fig_pays)

with tab2:  # Engagement & Fidélité
    st.header("Engagement et Fidélité")
    fig_produits = px.histogram(df_filtered, x="Produit bancaire", color="Attrition", barmode="overlay")
    st.plotly_chart(fig_produits)

    fig_carte = px.bar(df_filtered.groupby("Carte de crédit")["Attrition"].mean().reset_index(),
                       x="Carte de crédit", y="Attrition", title="Attrition selon Carte de Crédit")
    st.plotly_chart(fig_carte)

with tab3:  # Finance
    st.header("Facteurs Financiers")
    fig_salaire = px.histogram(df_filtered, x="Salaire", color="Attrition", barmode="overlay")
    st.plotly_chart(fig_salaire)

    fig_credit = px.box(df_filtered, x="Attrition", y="Score pour credit", title="Score de Crédit et Attrition")
    st.plotly_chart(fig_credit)

with tab4:  # Relation Client
    st.header("Relation Client")
    # Exemple : nombre d’interactions (besoin d’une colonne supplémentaire pour interactions)
    # fig_interactions = px.histogram(df_filtered, x="Nombre d'interactions", color="Attrition", barmode="overlay")
    # st.plotly_chart(fig_interactions)

with tab5:  # Prédictions & Machine Learning
    st.header("Modèle Prédictif - Risque d'Attrition")

    # Sélection des features et entraînement modèle
    features = ["Score pour credit", "Age", "Ancienneté", "Solde", "Produit bancaire", "Salaire"]
    X = df[features]
    y = df["Attrition"]
    model = RandomForestClassifier()
    # Remplacer les valeurs texte "oui"/"non" par 1/0 (exemple : 'Carte de crédit' et 'Membre actif')
    df["Carte de crédit"] = df["Carte de crédit"].map({"oui": 1, "non": 0})
    df["Membre actif"] = df["Membre actif"].map({"oui": 1, "non": 0})
    
    # Vérifier que toutes les colonnes utilisées sont bien numériques
    X = df[["Score pour credit", "Age", "Ancienneté", "Solde", "Produit bancaire", "Salaire"]].copy()
    X = X.apply(pd.to_numeric, errors="coerce")  # Convertir en nombres
    X = X.fillna(0)  # Remplacer les NaN par 0
    
    # Vérifier que y est bien binaire
    y = df["Attrition"].astype(int)

    model.fit(X, y)

    importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig_importance = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Importance des variables")
    st.plotly_chart(fig_importance)

    # Test de profil
    st.subheader("Testez le risque d'attrition d'un profil client")
    age_input = st.slider("Âge", int(df["Age"].min()), int(df["Age"].max()), 40)
    anciennete_input = st.slider("Ancienneté", int(df["Ancienneté"].min()), int(df["Ancienneté"].max()), 3)
    credit_input = st.slider("Score de Crédit", int(df["Score pour credit"].min()), int(df["Score pour credit"].max()), 600)
    solde_input = st.number_input("Solde du client", min_value=0, value=50000)
    produit_input = st.slider("Nombre de Produits Bancaires", 1, df["Produit bancaire"].max(), 2)
    salaire_input = st.number_input("Salaire annuel", min_value=0, value=50000)

    profil = np.array([[credit_input, age_input, anciennete_input, solde_input, produit_input, salaire_input]])
    prediction = model.predict_proba(profil)[0][1]

    st.write(f"**Risque d'attrition estimé : {prediction * 100:.2f}%**")

# ---- Lancer l'application ----
# Exécute : streamlit run app.py
