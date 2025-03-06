import streamlit as st
import pandas as pd
import plotly.express as px

# === CONFIGURATION DE LA PAGE ===
st.set_page_config(page_title="📊 Tableau de Bord SEO & Web Analytics", layout="wide")

# === CHARGEMENT DES DONNÉES ===
@st.cache_data
def load_data():
    file_path = "owa_action_fact2.csv"
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.columns = df.columns.str.strip()  # Nettoyer les noms de colonnes
    
    # Vérifier la présence des colonnes nécessaires
    required_columns = [
        'session_id', 'visitor_id', 'is_repeat_visitor', 'is_new_visitor',
        'numeric_value', 'days_since_prior_session', 'days_since_first_session',
        'action_name', 'action_group', 'medium', 'source_name'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not missing_columns:
        # Définition des poids pour le calcul du score d'engagement
        action_weights = {
            'frontend submit': 5,
            'frontend modify': 3,
            'editor publish': 7,
            'frontend create': 8,
            'view': 2
        }

        df['action_score'] = df['action_name'].map(action_weights).fillna(1)
        df['group_score'] = df['action_group'].apply(lambda x: 4 if x == 'publish' else 2)

        # Agréger les scores par visiteur
        df_grouped = df.groupby('visitor_id').agg(
            num_sessions=('session_id', 'nunique'),
            repeat_visitor=('is_repeat_visitor', 'max'),
            new_visitor=('is_new_visitor', 'max'),
            total_numeric_value=('numeric_value', 'sum'),
            avg_days_since_prior_session=('days_since_prior_session', 'mean'),
            avg_days_since_first_session=('days_since_first_session', 'mean'),
            total_action_score=('action_score', 'sum'),
            total_group_score=('group_score', 'sum'),
            unique_actions=('action_name', 'nunique'),
            unique_groups=('action_group', 'nunique')
        ).reset_index()

        # Assurer que certaines valeurs restent positives
        df_grouped['avg_days_since_first_session'] = df_grouped['avg_days_since_first_session'].apply(lambda x: max(x, 1))

        # Calcul du score d'engagement
        df_grouped['engagement_score'] = (
            df_grouped['num_sessions'] * 5 +
            df_grouped['repeat_visitor'] * 6 - df_grouped['new_visitor'] * 2 +
            df_grouped['total_numeric_value'] * 3 +
            (30 / (df_grouped['avg_days_since_prior_session'] + 1)) +
            (50 / (df_grouped['avg_days_since_first_session'] + 1)) +
            df_grouped['total_action_score'] * 2 +
            df_grouped['total_group_score'] * 3 +
            df_grouped['unique_actions'] * 4 +
            df_grouped['unique_groups'] * 5
        )

        # Normalisation entre 0 et 100
        df_grouped['engagement_score'] = (df_grouped['engagement_score'] - df_grouped['engagement_score'].min()) / (
            df_grouped['engagement_score'].max() - df_grouped['engagement_score'].min()) * 100

        df = df.merge(df_grouped[['visitor_id', 'engagement_score']], on='visitor_id', how='left')

    return df

df = load_data()

# === SIDEBAR (FILTRES DYNAMIQUES) ===
st.sidebar.header("🔍 Filtres")

# 🗓 Sélection de la période
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
start_date, end_date = st.sidebar.date_input("📆 Période :", [min_date, max_date], min_value=min_date, max_value=max_date)

filtered_df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]

# 🔗 Canaux d'acquisition
medium_selected = st.sidebar.multiselect("🛒 Canal d'acquisition", df["medium"].unique(), default=df["medium"].unique())

# 🔗 Sources
source_selected = st.sidebar.multiselect("🔗 Source", df["source_name"].dropna().unique(), default=df["source_name"].dropna().unique())

# 👥 Type de visiteur
visitor_type = st.sidebar.radio("👥 Type de visiteur", ["Tous", "Nouveau", "Récurrent"])

filtered_df = filtered_df[
    (filtered_df["medium"].isin(medium_selected)) & 
    (filtered_df["source_name"].isin(source_selected))
]

if visitor_type == "Nouveau":
    filtered_df = filtered_df[filtered_df["is_new_visitor"] == 1]
elif visitor_type == "Récurrent":
    filtered_df = filtered_df[filtered_df["is_repeat_visitor"] == 1]

# === CREATION DES ONGLETS ===
tabs = st.tabs(["🏠 Accueil", "📥 Acquisition", "🎭 Engagement", "🎯 Conversion & Rétention", "📊 Score d'Engagement", "🕒 Analyse Temporelle"])

# === 🏠 ACCUEIL (KPI GLOBAUX) ===
with tabs[0]:
    st.markdown("## 🏠 Vue Globale des Performances SEO")
    st.metric("👥 Sessions Totales", f"{filtered_df['session_id'].nunique():,}")
    st.metric("🧑‍💻 Visiteurs Uniques", f"{filtered_df['visitor_id'].nunique():,}")
    st.metric("🔁 Taux de Retour", f"{filtered_df['is_repeat_visitor'].mean()*100:.2f} %")

# === 📥 ACQUISITION ===
with tabs[1]:
    st.markdown("## 📥 Analyse du Trafic & Acquisition")
    medium_counts = filtered_df["medium"].value_counts().reset_index()
    medium_counts.columns = ["medium", "count"]
    fig_medium = px.bar(medium_counts, x="medium", y="count", title="📊 Canaux d'Acquisition")
    st.plotly_chart(fig_medium, use_container_width=True)

# === 🎭 ENGAGEMENT ===
with tabs[2]:
    st.markdown("## 🎭 Engagement Utilisateur")
    action_counts = filtered_df["action_name"].value_counts().reset_index()
    action_counts.columns = ["action_name", "count"]
    fig_actions = px.bar(action_counts.head(5), x="action_name", y="count", title="🔝 Top 5 Actions les Plus Réalisées")
    st.plotly_chart(fig_actions, use_container_width=True)

# === 🎯 CONVERSION & RÉTENTION ===
with tabs[3]:
    st.markdown("## 🎯 Conversion & Rétention")
    fig_conversion = px.bar(filtered_df.groupby("action_name")["session_id"].count().reset_index().sort_values(by="session_id", ascending=False).head(5), x="action_name", y="session_id", title="🎯 Actions Clés les Plus Convertissantes")
    st.plotly_chart(fig_conversion, use_container_width=True)

# === 📊 SCORE D’ENGAGEMENT ===
with tabs[4]:
    st.markdown("## 📊 Score d’Engagement des Visiteurs")
    filtered_df_eng = filtered_df[(filtered_df['engagement_score'] >= 0) & (filtered_df['engagement_score'] <= 20)]
    fig_engagement = px.scatter(filtered_df_eng, x='visitor_id', y='engagement_score', color='engagement_score', size='engagement_score', title="Engagement Score des Visiteurs")
    st.plotly_chart(fig_engagement, use_container_width=True)

# === 🕒 ANALYSE TEMPORELLE ===
with tabs[5]:
    st.markdown("## 🕒 Analyse Temporelle")
    fig_sessions_time = px.line(filtered_df.groupby("timestamp")["session_id"].count().reset_index(), x="timestamp", y="session_id", title="📅 Sessions par Jour")
    st.plotly_chart(fig_sessions_time, use_container_width=True)
