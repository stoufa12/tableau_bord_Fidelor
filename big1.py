import streamlit as st
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import locale
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from geopy.geocoders import Nominatim
import numpy as np
import pytz



st.set_page_config(page_title="Tableau de Bord - Fidelor", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
        body {
            background-color: #FFFFFF; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

def connect_to_database():
    try:
        return pymysql.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
    except pymysql.err.OperationalError as e:
        st.error("❌ Impossible de se connecter à la base de données. Vérifiez votre connexion Internet.")
        st.stop()
        
timezone = pytz.timezone("Africa/Dakar")
def now():
    return datetime.datetime.now(timezone)        

def get_contrats_classification():
    db_connection = connect_to_database()
    query = """
    SELECT c.id, c.client_id, cl.full_name AS Client, cl.phone AS Téléphone, cl.phone2 AS Téléphone_2, c.date_fin AS Date_fin, c.Montant_pret, frais AS Frais, DATEDIFF(c.date_fin, CURDATE()) AS Jours_restants
    FROM contract c
    JOIN client cl ON c.client_id = cl.id
    WHERE c.paiement = 'non payé' AND c.ajustement <> 1
    """
    df = pd.read_sql(query, db_connection)
    db_connection.close()

    df['classification'] = pd.cut(df['Jours_restants'],
                                  bins=[float('-inf'), -1, 7, 15, 21, 30, float('inf')],
                                  labels=['Échéance dépassé', 'Échéance dans les 7 jours', 'Échéance compris entre 7 et 15 jours', 'Échéance compris entre 15 et 21 jours', 'Échéance compris entre 21 et 30 jours', 'Échéance dans plus d\'un mois'])
    return df


def get_clients_by_city(table_name):
    db_connection = connect_to_database()
    query = f"""
    SELECT address, COUNT(*) AS nombre_clients
    FROM {table_name}
    GROUP BY address
    """
    df = pd.read_sql(query, db_connection)
    db_connection.close()
    return df

@st.cache_data(ttl = 604800)
def geocode_address(address, timeout=20):
    geolocator = Nominatim(user_agent="fidelor_app")
    location = geolocator.geocode(f"{address}, Dakar", timeout=timeout)
    if location:
        return location.latitude, location.longitude
    return None, None

def get_clients_growth(table_name):
    db_connection = connect_to_database()
    query = f"""
    SELECT DATE_FORMAT(reg_date, '%Y-%m') AS mois, COUNT(*) AS nombre_nouveaux_clients
    FROM {table_name}
    GROUP BY mois
    ORDER BY mois;
    """
    df = pd.read_sql(query, db_connection)
    db_connection.close()
    return df

def get_aor_data():
    db_connection = connect_to_database()
    query = f"""
    SELECT c.id, c.full_name AS client, COUNT(ct.id) AS nombre_contrats, SUM(ct.investissement + ct.fidelor_frais + ct.margePenalties) AS montant_total
    FROM client c
    JOIN contract ct ON c.id = ct.client_id
    WHERE ct.ajustement <> 1
    GROUP BY c.id
    """
    data = pd.read_sql(query, db_connection)
    db_connection.close()
    df = pd.DataFrame(data, columns=["client", "nombre_contrats", "montant_total"])
    df_sorted = df.sort_values(by = 'montant_total', ascending = False)
    df_top_client = df_sorted.head(7)
    return df_top_client 

def get_aoi_data():
    db_connection = connect_to_database()
    query = f"""
    SELECT cr.id, cr.full_name AS client, COUNT(br.id) AS nombre_contrats, SUM(br.prix_valoriser - br.prix_achat) AS montant_total
    FROM client_rachat cr
    JOIN bijou_achat br ON cr.id = br.client_id
    GROUP BY cr.id
    """
    data = pd.read_sql(query, db_connection)
    db_connection.close()
    df = pd.DataFrame(data, columns=["client", "nombre_contrats", "montant_total"])
    df_sorted = df.sort_values(by = 'montant_total', ascending = False)
    df_top_client = df_sorted.head(7)
    return df_top_client

st.sidebar.title("Navigation")
page = st.sidebar.radio("Suivi sur :", ["Contrats actifs", "Performance globale", "Investisseurs", "Fidelor", "Clientèle"])

def display_card(title, value, icon):
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 10px; padding: 20px; background-color: #f1f1f1; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin: 10px;">
            <h3 style="color: #000000;">{icon} {title}</h3>
            <h2 style="color: #333; font-weight: bold;">{value}</h2>
        </div>
    """, unsafe_allow_html=True)

if page == "Contrats actifs":
    st.title("📑 Contrats actifs")

    col1, col2 = st.columns(2)

    df_contrats = get_contrats_classification()

    nombre_contrats_actifs = len(df_contrats)
    with col1:
        display_card("Contrats actifs", nombre_contrats_actifs, "📄")

    contrat_counts = df_contrats['classification'].value_counts().reset_index()
    contrat_counts.columns = ['Classification', 'Nombre']

    fig = px.bar(contrat_counts, x='Classification', y='Nombre', color='Classification')
    fig.update_traces(text=contrat_counts['Nombre'].astype(str), textposition='outside', texttemplate='%{y}', showlegend=False)

    st.subheader("📊 Répartition du nombre de contrats actifs")
    st.plotly_chart(fig, use_container_width=True)
    
    def calcul_frais(group):
        if group.name == 'Échéance dépassé':
            return (group["Frais"] + (group["Frais"] * abs(group["Jours_restants"]) * 0.004)).sum()
        else:
            return group["Frais"].sum()

    def calcul_total(group):
        if group.name == 'Échéance dépassé':
            return (group["Frais"] + (group["Frais"] * abs(group["Jours_restants"]) * 0.004) + group["Montant_pret"]).sum()
        else:
            return (group["Frais"] + group["Montant_pret"]).sum()
            
    
    frais_par_classification = df_contrats.groupby("classification").apply(calcul_frais).reset_index(name="Frais Total")
    total_par_classification = df_contrats.groupby("classification").apply(calcul_total).reset_index(name="Total")
    # Rejoindre avec le DataFrame resume
    resume = df_contrats.groupby("classification")["Montant_pret"].sum().reset_index() 
    resume.columns = ["Classification selon l\'échéance", "Montant (sans les frais de garde)"]
    
    resume = resume.merge(frais_par_classification, left_on="Classification selon l\'échéance", right_on="classification")
    resume.drop(columns="classification", inplace=True)

    resume = resume.merge(total_par_classification, left_on="Classification selon l\'échéance", right_on="classification")
    resume.drop(columns="classification", inplace=True)
    
    resume = resume[["Classification selon l\'échéance", "Montant (sans les frais de garde)", "Frais Total", "Total"]]
    valeur_interne = resume["Montant (sans les frais de garde)"].sum()
    st.subheader("📋 Répartition montant à récupérer dans les prochains jours")
    st.dataframe(
        resume.style.format({
            "Montant (sans les frais de garde)": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA", 
            "Frais Total": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA",
            "Total": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA"
        }), 
        use_container_width=True,
        hide_index=True
    )

    
    st.subheader("⏱ Contrats actifs en retard paiement")
    df_retard = df_contrats[df_contrats['classification'] == 'Échéance dépassé']
    nombre_contrat_dépassé = len(df_retard)
    with col2:
        display_card("Retard paiement", nombre_contrat_dépassé , "📄")
        
    df_retard["Jours de retard"] = df_retard["Jours_restants"].abs() 
    # Colonne Pénalités = frais * 0,004 * jours de retard
    df_retard["Pénalités"] = df_retard["Montant_pret"] * 0.004 * df_retard["Jours de retard"]
    df_retard = df_retard[['Client', 'Téléphone', 'Montant_pret', 'Frais', 'Pénalités', 'Date_fin', 'Jours de retard']]
    
    if not df_retard.empty:
        max_jours_retard = int(df_retard['Jours de retard'].max())
        min_retard, max_retard = st.slider("Filtrer par jours de retard", 0, max_jours_retard, (0, max_jours_retard))

        df_retard_filtre = df_retard[df_retard['Jours de retard'].between(min_retard, max_retard)]

        st.dataframe(
        df_retard_filtre.style.format({
            "Montant_pret": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA", 
            "Frais": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA",
            "Pénalités": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA"
        }), 
        use_container_width=True,
        hide_index=True
    )
    else:
        st.info("Aucun contrat en retard.")

    df_anomalies = df_retard_filtre[df_retard_filtre["Jours de retard"] > 15]

    if not df_anomalies.empty:
        lignes = ""
        for _, row in df_anomalies.iterrows():
            client = row["Client"]
            montant = format(row["Montant_pret"], ",.0f").replace(",", " ")
            lignes += f"<div>- <b>{client}</b> pour le contrat de <b>{montant}</b> FCFA</div>"
        st.markdown(f"""
        <div style='background-color:#ffe6e6; padding:10px; border-left:6px solid red;'>
            <b style='color:red;'>⚠️ Attention :</b> Les clients suivants ont un retard de paiement de <b>plus de 15 jours</b> :<br><br>
            <span style='color:black;'>{lignes}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💎 Estimation des bijoux actifs")    
    db_connection = connect_to_database()
    query = """
        SELECT 
            b.carats,
            b.importer,
            SUM(b.poids) AS total_poids,
            SUM(b.poids * p.prix) AS total_valeur_marche
        FROM bijou b
        JOIN contract c ON b.contract_id = c.id
        JOIN prix_or p 
            ON b.carats = p.type AND b.importer = p.provenance
        WHERE c.paiement = 'non payé' AND ajustement <> 1
        GROUP BY b.carats, b.importer;
        
    """
    df = pd.read_sql(query, db_connection)
    db_connection.close()
    df.columns = ['Type', 'Provenance', 'Poids(g)', 'Valeur marché']
    valeur_marche = df['Valeur marché'].sum()
    difference = valeur_marche - valeur_interne
    valeur_interne_fmt = "{:,.0f}".format(valeur_interne).replace(",", " ")
    valeur_marche_fmt = "{:,.0f}".format(valeur_marche).replace(",", " ")

    if difference > 0:
        color = "#28a745"  # vert
        message = f"Sous-évalué de {'{:,.0f}'.format(difference).replace(',', ' ')} FCFA"
    elif difference < 0:
        color = "#dc3545"  # rouge
        message = f"Surévalué de {'{:,.0f}'.format(abs(difference)).replace(',', ' ')} FCFA"
    else:
        color = "#ffc107"  # jaune
        message = "Évaluation équilibrée"

    st.markdown(f"""
    <div style="
        display: flex;
        border: 1px solid #ccc;
        border-left: 10px solid {color};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        font-family: Arial, sans-serif;
    ">
        <div>
            <h4 style="margin-top: 0;">⚖️ Valeur du stock</h4>
            <p style="margin: 6px 0;"><strong>Valeur interne :</strong> {valeur_interne_fmt} FCFA</p>
            <p style="margin: 6px 0;"><strong>Valeur marché :</strong> {valeur_marche_fmt} FCFA</p>
            <p style="margin-top: 10px; font-weight: bold; color: {color};">{message}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df.style.format({
            "Poids(g)": lambda x: f"{x:,.2f}".replace(",", " ") + " g", 
            "Valeur marché": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA"
        }), 
        use_container_width=True,
        hide_index=True
    )
    
elif page == "Clientèle":
    st.title("👥 Clients")

    col1, col2 = st.columns(2)

    df_clients_aoi = get_clients_by_city('client_rachat')
    df_clients_aor = get_clients_by_city('client')

    with col1:
        display_card("Nombre de clients AOI", df_clients_aoi['nombre_clients'].sum(), "👤")

    with col2:
        display_card("Nombre de clients AOR", df_clients_aor['nombre_clients'].sum(), "👤")

    st.subheader("📈 Croissance du nombre de nouveaux clients")
    filtre_type_client = st.radio("Sélectionner le type d'activité", ["Achat réméré", "Achat immédiat"], horizontal=True)

    table_name = "client" if filtre_type_client == "Achat réméré" else "client_rachat"
    df_growth = get_clients_growth(table_name)

    # Créer un graphique
    fig_bar = px.bar(df_growth, 
                 x='mois', 
                 y='nombre_nouveaux_clients', 
                 title="Évolution du nombre de nouveaux clients par mois", 
                 labels={'mois': 'Mois', 'nombre_nouveaux_clients': 'Nombre de nouveaux clients'})

    fig_bar.update_traces(text=df_growth['nombre_nouveaux_clients'].astype(str), textposition='outside', texttemplate='%{y}')

    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("♻️ Valeur à vie des meilleurs clients")
    filtre_type_client =st.radio("Sélectionnez le type d'activité", ["Achat réméré", "Achat immédiat"], horizontal = True)
    if filtre_type_client == "Achat réméré":    
         df = get_aor_data()
    elif filtre_type_client == "Achat immédiat":
         df = get_aoi_data()
    if df.empty:
        st.warning("Aucune donnée disponible pour cette sélection.")
    else:
        heatmap_data = df.pivot_table(index='client', columns='nombre_contrats', values='montant_total', fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title(f"Top 7 des clients avec les plus-values en FCFA ({filtre_type_client}) ")
        ax.set_xlabel("Nombre de Contrats")
        ax.set_ylabel("Nom Client")

    st.pyplot(fig)
    
    st.subheader("🌍 Répartition géographique de nos clients")

    filtre_type_client = st.radio("Type client", ["Achat réméré", "Achat immédiat"], horizontal=True)

    df_map = df_clients_aor if filtre_type_client == "Achat réméré" else df_clients_aoi
    df_map['latitude'], df_map['longitude'] = zip(*df_map['address'].apply(geocode_address))

    df_map_valid = df_map.dropna(subset=['latitude', 'longitude'])
    fig_map = px.scatter_mapbox(df_map_valid, lat='latitude', lon='longitude', size='nombre_clients', color='nombre_clients', color_continuous_scale= 'RdBu', hover_name='address', zoom=10, mapbox_style="open-street-map")

    st.plotly_chart(fig_map, use_container_width=True)
    
elif page == "Performance globale":
    st.title("🔄 Performance globale par type d'achat")

    st.subheader("Indicateurs financiers clé pour l'achat réméré")
    
    col1, col2 = st.columns(2)    
    with col1:
        selected_year = st.selectbox(
           "Sélectionner l'année", 
            list(range(2022, now().year + 1))[::-1],
            key="annee_réméré"
        )

    mois_fr = [
         "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
         "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]

    current_month = now().month

    with col2:
        selected_month = st.selectbox(
            "Sélectionner le mois",
            options=list(range(1, 13)),
            index=current_month - 1,
            format_func=lambda x: mois_fr[x - 1],
            key="mois_réméré"
        )


    db_connection = connect_to_database()
    query = """
        SELECT montant_pret AS capital, investissement,  fidelor_frais, renouvellement
        FROM contract
        WHERE MONTH(date_paye) = %s AND YEAR(date_paye) = %s AND paiement <> 'non payé' AND ajustement <> 1
    """
    
    cursor = db_connection.cursor()
    cursor.execute(query, (selected_month, selected_year))
    rows = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df = pd.DataFrame(rows, columns=["capital", "investissement", "fidelor_frais", "renouvellement"])

    if not df.empty:
        capital_injecte = df["capital"].sum()
        capital_renouvele = df[df["renouvellement"] != 0]["capital"].sum()
        interets_generes = (df["investissement"] + df["fidelor_frais"] ).sum()
        capital_ayant_genere = capital_injecte - capital_renouvele

        taux_rendement = (interets_generes / capital_injecte) * 100 if capital_injecte > 0 else 0
        taux_rotation = (capital_ayant_genere / capital_injecte) * 100 if capital_injecte > 0 else 0
        taux_rentabilite = (interets_generes / capital_ayant_genere) * 100 if capital_ayant_genere > 0 else 0
    else:
        taux_rendement = taux_rotation = taux_rentabilite = 0
    
    
    col1, col2 = st.columns(2)

    with col1:
        display_card("Rendement de l'activité (%)", round(taux_rendement, 2), "💰")

    with col2:
        display_card("Rentabilité de l'investissement (%)", round(taux_rentabilite, 2), "📈")

    db_connection = connect_to_database()
    query_count = """
        SELECT COUNT(*) FROM contract
        WHERE MONTH(reg_date) = %s AND YEAR(reg_date) = %s AND ajustement <> 1 AND renouvellement = 0
         
    """
    cursor = db_connection.cursor()
    cursor.execute(query_count, (selected_month, selected_year))
    nb_renouvellements = cursor.fetchone()[0]
    cursor.close()
    db_connection.close()

    db_connection = connect_to_database()
    query_count_1 = """
        SELECT COUNT(*) FROM contract
        WHERE MONTH(reg_date) = %s AND YEAR(reg_date) = %s AND ajustement <> 1 AND renouvellement <> 0
    """
    cursor = db_connection.cursor()
    cursor.execute(query_count_1, (selected_month, selected_year))
    nb_renouvellements_1 = cursor.fetchone()[0]
    cursor.close()
    db_connection.close()
    
    db_connection = connect_to_database()
    query_count_2 = """
        SELECT COUNT(*) FROM contract
        WHERE MONTH(date_paye) = %s AND YEAR(date_paye) = %s AND ajustement <> 1 AND solde = 0
       
    """
    cursor = db_connection.cursor()
    cursor.execute(query_count_2, (selected_month, selected_year))
    nb_renouvel_2 = cursor.fetchone()[0]
    cursor.close()
    db_connection.close()
 
    col1, col2 = st.columns(2)

    with col1:
        display_card("Contrats reconduits", nb_renouvellements_1, "↩️") 

    with col2:
        display_card("Rotation des fonds (%)", round(taux_rotation, 2), "🔄")
    
    col1, col2 = st.columns(2)

    with col1:
        display_card("Nouveaux engagements", nb_renouvellements, "🆕") 

    with col2:
        display_card("Sorties définitives (liquidations)", nb_renouvel_2, "🔚")

    st.markdown("### 📊 Tableau annuel de l'achat réméré")
      
    years = list(range(2022, now().year + 1))
    current_year = now().year

    selected_year = st.selectbox(
       "Sélectionner l'année pour le tableau de synthèse",
        years,
        index=years.index(current_year)
    )

    db_connection = connect_to_database()
    query_paye = """
        SELECT 
            MONTH(date_paye) AS mois,
            SUM(montant_pret) AS capital_total,
            SUM(CASE WHEN renouvellement = 0 THEN montant_pret ELSE 0 END) AS capital_generateur,
            SUM(investissement + fidelor_frais) AS interets_gen
        FROM contract
        WHERE YEAR(date_paye) = %s
        AND paiement <> 'non payé' AND ajustement <> 1
        GROUP BY mois
        ORDER BY mois
     """
    cursor = db_connection.cursor()
    cursor.execute(query_paye, (selected_year,))
    rows_paye = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df_paye = pd.DataFrame(rows_paye, columns=[
         "Mois", "Capital Injecté", "Capital Générateur", "Intérêts Générés"
    ])
    

    db_connection = connect_to_database()
    query_non_paye = """
        SELECT 
            MONTH(reg_date) AS Mois,
            COUNT(*) AS Nouveaux_renouvellements
        FROM contract
        WHERE YEAR(reg_date) = %s
        AND renouvellement <> 0 AND ajustement <> 1
        GROUP BY mois
        ORDER BY mois;
    """
    cursor = db_connection.cursor()
    cursor.execute(query_non_paye, (selected_year,))
    rows_non_paye = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df_non_paye = pd.DataFrame(rows_non_paye, columns=["Mois", "Renouvellements"])

    df = pd.merge(df_paye, df_non_paye, on="Mois", how="left")
    df.fillna(0, inplace=True)
    df_all_aor = df
    cols = ["Capital Injecté", "Capital Générateur", "Intérêts Générés", "Renouvellements"]
    df[cols] = df[cols].apply(pd.to_numeric, errors = 'coerce')

    df["Taux Rendement Activité"] = (df["Intérêts Générés"] / df["Capital Injecté"]) * 100
    df["Taux Rentabilité Investissement"] = (df["Intérêts Générés"] / df["Capital Générateur"]) * 100
    df["Taux Rotation"] = (df["Capital Générateur"] / df["Capital Injecté"]) * 100
    df = df.round(2)

    mois_fr = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
     ]

    df["Mois"] = df["Mois"].apply(lambda x: mois_fr[int(x) - 1] if pd.notna(x) else x)

    def evolution_flèche(colonne):
        flèches = []
        for i in range(len(df)):
            if i == 0:
                flèches.append("")
            else:
                if pd.isna(df[colonne].iloc[i]) or pd.isna(df[colonne].iloc[i-1]):
                     flèches.append("")
                elif df[colonne].iloc[i] > df[colonne].iloc[i-1]:
                     flèches.append("🟩 ↑")
                elif df[colonne].iloc[i] < df[colonne].iloc[i-1]:
                     flèches.append("🟥 ↓")
                else:
                     flèches.append("⬜ =")
        return flèches
    df["Flèche Rendement"] = evolution_flèche("Taux Rendement Activité")
    df["Flèche Rentabilité"] = evolution_flèche("Taux Rentabilité Investissement")
    df["Flèche Rotation"] = evolution_flèche("Taux Rotation")

    df["Rendement Activité"] = df["Taux Rendement Activité"].astype(str) + " % " + df["Flèche Rendement"]
    df["Rentabilité Investissement"] = df["Taux Rentabilité Investissement"].astype(str) + " % " + df["Flèche Rentabilité"]
    df["Rotation Fonds"] = df["Taux Rotation"].astype(str) + " % " + df["Flèche Rotation"]

    df_final = df[["Mois", "Rendement Activité", "Rentabilité Investissement", "Rotation Fonds",  "Renouvellements"]]
    st.dataframe(df_final, use_container_width=True, hide_index=True)

    st.markdown("### 📊 Historique des tendances de paiement de l'achat réméré")
    db_connection = connect_to_database()
    query = """
        SELECT 
            DATE_FORMAT(date_paye, '%Y-%m') AS periode,
            paiement,
            COUNT(*) AS total
        FROM contract
        WHERE date_paye IS NOT NULL AND paiement <> 'non payé'
        GROUP BY periode, paiement
        ORDER BY periode ASC;
        
    """
    df_type = pd.read_sql(query, db_connection)
    db_connection.close()
    df_type.columns = ['Période', 'Mode paiement', 'Total']
    mapping = {
        'Contrat réglé par anticipation' : 'Anticipé',
        'Contrat réglé à échéance' : 'À échéance',
        'Contrat réglé Post-Echeance' : 'Après écheance',
        'Contrat en défaut de paiement' : 'Défaut paiement',
        'Contrat en défaut racheté' : 'Défaut racheté'
    }

    df_type['Mode paiement'] = df_type['Mode paiement'].replace(mapping)

    df_type['Période'] = pd.to_datetime(df_type['Période'], format='%Y-%m')
    df_type['Affichage'] = df_type['Période'].dt.strftime('%b %Y')  # Ex: "Jan 2023"

    df_type['Affichage'] = pd.Categorical(
        df_type['Affichage'],
        categories=sorted(df_type['Affichage'].unique(), key=lambda x: pd.to_datetime(x, format='%b %Y')),
        ordered=True
    )

    chart = alt.Chart(df_type).mark_bar().encode(
        x=alt.X('Affichage:N', title='Période', sort=list(df_type['Affichage'].unique())),
        y=alt.Y('Total:Q', title='Nombre de contrats'),
        color=alt.Color('Mode paiement:N', title='Type de paiement'),
        tooltip=['Affichage', 'Mode paiement', 'Total']
    ).properties(
        width=850,
        height=400,
        title='Répartition mensuelle des paiements par type de paiement'
    )

    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("Indicateurs financiers clé pour l'achat immédiat") 
    col1, col2 = st.columns(2)    
    with col1:
        selected_year = st.selectbox(
           "Sélectionner l'année", 
            list(range(2022, now().year + 1))[::-1],
            key="annee_immédiat"
        )

    mois_fr = [
         "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
         "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]

    current_month = now().month

    with col2:
        selected_month = st.selectbox(
            "Sélectionner le mois",
            options=list(range(1, 13)),
            index=current_month - 1,
            format_func=lambda x: mois_fr[x - 1],
            key="mois_immédiat"
        )

    db_connection = connect_to_database()
    query_immédiat = """
        SELECT 
          prix_achat AS capital, 
          investissement,  
          CASE 
            WHEN prix_valoriser <> 0 THEN fidelor 
            ELSE 0 
          END AS fidelor
        FROM bijou_achat
        WHERE MONTH(reg_date) = %s AND YEAR(reg_date) = %s;
    """
    
    cursor = db_connection.cursor()
    cursor.execute(query_immédiat, (selected_month, selected_year))
    rows_immédiat = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df_immédiat = pd.DataFrame(rows_immédiat, columns=["capital", "investissement", "fidelor"] )
    db_connection = connect_to_database()
    query_compte = """
        SELECT COUNT(*)
        FROM bijou_achat
        WHERE MONTH(reg_date) = %s AND YEAR(reg_date) = %s 
    """
    
    cursor = db_connection.cursor()
    cursor.execute(query_compte, (selected_month, selected_year))
    nb_achat_immédiat = cursor.fetchone()[0]
    cursor.close()
    db_connection.close()
    if not df_immédiat.empty:
        taux_rentabilite = ((df_immédiat["investissement"] + df_immédiat["fidelor"] ).sum()/ df_immédiat["capital"].sum()) * 100 if df_immédiat["capital"].sum() > 0 else 0  
    
    else:
        taux_rentabilite = 0
    col1, col2 = st.columns(2)

    with col1:
        display_card("Rentabilité de l'activité (%)", round(taux_rentabilite, 2), "💰")

    with col2:
        display_card("Nombre achat immédiat", nb_achat_immédiat, "📈")

    st.markdown("### 📊 Tableau annuel de l'achat immédiat")
    # Sélecteur d’année
    years = list(range(2022, now().year + 1))
    current_year = now().year

    selected_year = st.selectbox(
       "Sélectionner l'année pour le tableau de synthèse",
        years,
        index=years.index(current_year),
        key="annee_immédiat_1"
    )

    db_connection = connect_to_database()
    query_immédiat = """
        SELECT 
            MONTH(reg_date) AS mois,
            SUM(prix_achat) AS capital,
            COUNT(*) AS nombre_achat,
            SUM(investissement) + SUM(CASE WHEN prix_valoriser <> 0 THEN fidelor ELSE 0 END) AS interets
        FROM bijou_achat
        WHERE YEAR(reg_date) = %s
        GROUP BY mois
        ORDER BY mois
     """
    cursor = db_connection.cursor()
    cursor.execute(query_immédiat, (selected_year,))
    rows_immédiat = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df_immédiat = pd.DataFrame(rows_immédiat, columns=[
         "Mois", "capital", "nombre_achat", "interets"
    ])
    df_immédiat.fillna(0, inplace=True)
    df_all_aoi = df_immédiat
    cols = ["capital", "nombre_achat", "interets"]
    df_immédiat[cols] = df_immédiat[cols].apply(pd.to_numeric, errors = 'coerce')

    # Calcul des taux
    df_immédiat["Taux Rentabilité Investissement"] = (df_immédiat["interets"] / df_immédiat["capital"]) * 100
    df_immédiat["Taux Rotation"] = (df_immédiat["capital"] / df_immédiat["capital"]) * 100
    df_immédiat = df_immédiat.round(2)
    
    mois_fr = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
     ]

    df_immédiat["Mois"] = df_immédiat["Mois"].apply(lambda x: mois_fr[int(x) - 1] if pd.notna(x) else x)

    def evolution_flèche(colonne):
        flèches = []
        for i in range(len(df_immédiat)):
            if i == 0:
                flèches.append("")
            else:
                if pd.isna(df_immédiat[colonne].iloc[i]) or pd.isna(df_immédiat[colonne].iloc[i-1]):
                     flèches.append("")
                elif df_immédiat[colonne].iloc[i] > df_immédiat[colonne].iloc[i-1]:
                     flèches.append("🟩 ↑")
                elif df_immédiat[colonne].iloc[i] < df_immédiat[colonne].iloc[i-1]:
                     flèches.append("🟥 ↓")
                else:
                     flèches.append("⬜ =")
        return flèches

    df_immédiat["Flèche Rentabilité"] = evolution_flèche("Taux Rentabilité Investissement")
    df_immédiat["Flèche Rotation"] = evolution_flèche("Taux Rotation")
    # Fusion des colonnes avec flèches
    df_immédiat["Rentabilité Investissement"] = df_immédiat["Taux Rentabilité Investissement"].astype(str) + " % " + df_immédiat["Flèche Rentabilité"]
    df_immédiat["Rotation Fonds"] = df_immédiat["Taux Rotation"].astype(str) + " % "

    df_final_immédiat = df_immédiat[["Mois", "Rentabilité Investissement", "Rotation Fonds"]]
    # Affichage
    st.dataframe(df_final_immédiat, use_container_width=True, hide_index=True)
    
    st.markdown("### 📊 Tableau synthèse annuel de l'achat immédiat + réméré")
 
    years = list(range(2022, now().year + 1))
    current_year = now().year

    selected_year = st.selectbox(
       "Sélectionner l'année pour le tableau de synthèse",
        years,
        index=years.index(current_year),
        key="annee_all"
    )

    db_connection = connect_to_database()
    query_all = """
        SELECT 
        mois,
        SUM(capital) AS total_capital,
        SUM(interet) AS total_interet
        FROM (
        SELECT 
            MONTH(reg_date) AS mois,
            prix_achat AS capital,
            investissement + 
            CASE 
              WHEN prix_valoriser <> 0 THEN fidelor 
              ELSE 0 
            END AS interet
        FROM bijou_achat
        WHERE YEAR(reg_date) = %s

        UNION ALL

        SELECT 
            MONTH(date_paye) AS mois,
            montant_pret AS capital,
            investissement + fidelor_frais AS interet
        FROM contract
        WHERE YEAR(date_paye) = %s
        AND paiement <> 'non payé'
        AND ajustement <> 1
        ) AS union_mois
        GROUP BY mois
        ORDER BY mois
    """
    cursor = db_connection.cursor()
    cursor.execute(query_all, (selected_year, selected_year))
    rows_all = cursor.fetchall()
    cursor.close()
    db_connection.close()
    df_rentabilite = pd.DataFrame(rows_all, columns=["mois", "Capital Générateur", "Intérêts"])
    df_rentabilite.fillna(0, inplace=True)
    cols = ["Capital Générateur", "Intérêts"]
    df_rentabilite[cols] = df_rentabilite[cols].apply(pd.to_numeric, errors = 'coerce')

    df_rentabilite["Rentabilité de l'activité en %"] = (df_rentabilite["Intérêts"] / df_rentabilite["Capital Générateur"]) * 100  
    df_rentabilite = df_rentabilite.round(2)
    
    mois_fr = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
     ]
    df_rentabilite["Mois"] = df_rentabilite["mois"].apply(lambda x: mois_fr[int(x) - 1] if pd.notna(x) else x)
    # Colonnes finales
    df_rentabilite= df_rentabilite[["Mois", "Capital Générateur", "Intérêts", "Rentabilité de l'activité en %"]]
    df_rentabilite["Rentabilité de l'activité en %"] = df_rentabilite["Rentabilité de l'activité en %"].apply(
        lambda x: f"{x:.2f}".replace(".", ",") if pd.notnull(x) else ""
    )
    st.dataframe(
        df_rentabilite.style.format({
            "Capital Générateur": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA", 
            "Intérêts": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA",
           
        }), 
        use_container_width=True,
        hide_index=True
    )

    fig = px.pie(
        df_rentabilite,
        names="Mois",
        values="Intérêts",
        title=f"Répartition mensuelle des intérêts – Année {selected_year}",
        color_discrete_sequence=px.colors.qualitative.Vivid  # Palette personnalisée
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "Investisseurs":
    st.title("💸 Investissement")

    db_connection = connect_to_database()
    query_soldes_normaux = """
    SELECT 
        m.id,
        m.full_name AS Investisseur,
        COALESCE(c.total_achat_remere, 0)
        + COALESCE(d.total_achat_immediat, 0)
        + COALESCE(b.total_bonus, 0)
        + COALESCE(t.total_depot, 0)
        - COALESCE(t.total_retrait, 0) AS solde
    FROM mutualiseur m

    LEFT JOIN (
        SELECT mutualiseur, SUM(CASE WHEN ajustement <> 1 THEN investissement ELSE 0 END) AS total_achat_remere
        FROM contract
        GROUP BY mutualiseur
    ) c ON c.mutualiseur = m.id

    LEFT JOIN (
        SELECT mutualiseur, SUM(investissement) AS total_achat_immediat
        FROM bijou_achat
        GROUP BY mutualiseur
    ) d ON d.mutualiseur = m.id

    LEFT JOIN (
        SELECT mutualiseur, SUM(montant) AS total_bonus
        FROM bonus
        GROUP BY mutualiseur
    ) b ON b.mutualiseur = m.id

    LEFT JOIN (
        SELECT 
            mutualiseur_id,
            SUM(CASE WHEN type = 'depot' THEN montant ELSE 0 END) AS total_depot,
            SUM(CASE WHEN type = 'retrait' THEN montant ELSE 0 END) AS total_retrait
        FROM transactions_m
        GROUP BY mutualiseur_id
    ) t ON t.mutualiseur_id = m.id

    WHERE m.id NOT IN (0, 1)

    """
    df_solde_normaux = pd.read_sql(query_soldes_normaux, db_connection)
    db_connection.close()

    db_connection = connect_to_database()
    query_solde_special = """
    SELECT 
        m.id,
        m.full_name AS Investisseur,
        COALESCE(c.total_achat_remere, 0)
        + COALESCE(d.total_achat_immediat, 0)
        + COALESCE(b.total_bonus, 0)
        + COALESCE(t.total_depot, 0)
        - COALESCE(t.total_retrait, 0) AS solde
    FROM mutualiseur m

    LEFT JOIN (
        SELECT mutualiseur, 
       SUM(CASE WHEN ajustement <> 1 THEN investissement ELSE 0 END) AS total_achat_remere
       FROM contract
       WHERE date_paye > '2025-03-13'
       GROUP BY mutualiseur
    ) c ON c.mutualiseur = m.id

    LEFT JOIN (
        SELECT mutualiseur, SUM(investissement) AS total_achat_immediat
        FROM bijou_achat
        GROUP BY mutualiseur
    ) d ON d.mutualiseur = m.id

    LEFT JOIN (
        SELECT mutualiseur, SUM(montant) AS total_bonus
        FROM bonus
        GROUP BY mutualiseur
    ) b ON b.mutualiseur = m.id

    LEFT JOIN (
        SELECT 
            mutualiseur_id,
            SUM(CASE WHEN type = 'depot' THEN montant ELSE 0 END) AS total_depot,
            SUM(CASE WHEN type = 'retrait' THEN montant ELSE 0 END) AS total_retrait
        FROM transactions_m
        GROUP BY mutualiseur_id
    ) t ON t.mutualiseur_id = m.id

    WHERE m.id = 1

    """
    df_solde_special = pd.read_sql(query_solde_special, db_connection)
    db_connection.close()

    db_connection = connect_to_database()
    query_soldes_promodor = """
    SELECT 
        m.id + 100 AS id,
        m.full_name AS Investisseur,
        COALESCE(c.total_achat_remere, 0)
        + COALESCE(d.total_achat_immediat, 0)
        + COALESCE(b.total_bonus, 0)
        + COALESCE(t.total_depot, 0)
        - COALESCE(t.total_retrait, 0) AS solde
    FROM promotteur m

    LEFT JOIN (
        SELECT promotteur, SUM(CASE WHEN ajustement <> 1 THEN investissement ELSE 0 END) AS total_achat_remere
        FROM contract
        GROUP BY promotteur
    ) c ON c.promotteur = m.id

    LEFT JOIN (
        SELECT promotteur, SUM(investissement) AS total_achat_immediat
        FROM bijou_achat
        GROUP BY promotteur
    ) d ON d.promotteur = m.id

    LEFT JOIN (
        SELECT promotteur, SUM(montant) AS total_bonus
        FROM bonus
        GROUP BY promotteur
    ) b ON b.promotteur = m.id

    LEFT JOIN (
        SELECT 
            promotteur,
            SUM(CASE WHEN type = 'depot' THEN montant ELSE 0 END) AS total_depot,
            SUM(CASE WHEN type = 'retrait' THEN montant ELSE 0 END) AS total_retrait
        FROM transactions_p
        GROUP BY promotteur
    ) t ON t.promotteur= m.id

    WHERE m.id <> 0

    """
    df_solde_promodor = pd.read_sql(query_soldes_promodor, db_connection)
    db_connection.close()

    db_connection = connect_to_database()
    query_encours_1 = """
    SELECT 
        mutualiseur,
        SUM(montant_pret) AS montant_encours
        FROM contract
        WHERE paiement = 'non payé' AND ajustement <> 1 
        GROUP BY mutualiseur
    """
    df_encours_1 = pd.read_sql(query_encours_1, db_connection)
    db_connection.close()
    db_connection = connect_to_database()
    query_encours_2 = """
    SELECT 
        promotteur + 100 AS mutualiseur,
        SUM(montant_pret) AS montant_encours
        FROM contract
        WHERE paiement = 'non payé' AND ajustement <> 1 AND promotteur <> 0
        GROUP BY promotteur
    """
    df_encours_2 = pd.read_sql(query_encours_2, db_connection)
    db_connection.close()
    

    df_encours = pd.concat([df_encours_1[['mutualiseur', 'montant_encours']], 
                       df_encours_2[['mutualiseur', 'montant_encours']]], ignore_index=True)

    df_solde_normaux['id_mutualiseur'] = df_solde_normaux['id'] 
    df_solde_special['id_mutualiseur'] = df_solde_special['id']  
    df_solde_promodor['id_mutualiseur'] = df_solde_promodor['id']  

    df_soldes = pd.concat([df_solde_normaux[['id_mutualiseur', 'Investisseur', 'solde']], 
                       df_solde_promodor[['id_mutualiseur', 'Investisseur', 'solde']],
                       df_solde_special[['id_mutualiseur', 'Investisseur', 'solde']]], ignore_index=True)

    df_complet = df_soldes.merge(df_encours, left_on="id_mutualiseur", right_on="mutualiseur", how="left")

    df_complet["montant_encours"] = df_complet["montant_encours"].fillna(0)

    df_complet["Montant dispo"] = df_complet["solde"] - df_complet["montant_encours"]

    df_complet["Taux_liquidite"] = (df_complet["Montant dispo"] / df_complet["solde"]) * 100
    df_complet["Taux_liquidite"] = df_complet["Taux_liquidite"].round(2)

    liquidite_globale = ((df_complet["Montant dispo"].sum() / df_complet["solde"].sum()) * 100)
    montant_dispo_global = (df_complet["Montant dispo"].sum())

    df_taux_liquidite = df_complet[["Investisseur", "Taux_liquidite"]].sort_values(by="Taux_liquidite", ascending=False)

    df_montants_dispo = df_complet[["Investisseur", "Montant dispo"]].sort_values(by="Montant dispo", ascending=False)

    col1, col2 = st.columns(2)

    montant_formatte = f"{round(montant_dispo_global):,}".replace(",", " ") + " FCFA"
    pourcentage_formatte = f"{liquidite_globale:.2f} %"
    with col1:
        display_card("Montant global dispo", montant_formatte, "💰")

    with col2:
        display_card("Liquidité globale", pourcentage_formatte, "💰")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Montant dispo par investisseur")
        st.dataframe(
        df_montants_dispo.style.format({
            "Montant dispo": lambda x: f"{x:,.0f}".replace(",", " ") + " FCFA"
            }), use_container_width=True, hide_index=True
        )

    with col2:
        st.subheader("Liquidité en % par investisseur")
        st.dataframe(df_taux_liquidite, use_container_width=True, hide_index=True)  
        
    df_anomalies = df_montants_dispo[df_montants_dispo["Montant dispo"] < 0]

    if not df_anomalies.empty:
        noms = noms = "<br>".join(f"- {nom}" for nom in df_anomalies["Investisseur"])
        st.markdown(f"""
        <div style='background-color:#ffe6e6; padding:10px; border-left:6px solid red;'>
            <b style='color:red;'>⚠️ Attention :</b> Les investisseurs suivants ont un <b>montant disponible négatif</b> :<br>
            <span style='color:black;'>{noms}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Flux financiers liés à l'investissement")
    def get_flux_data(selected_year, selected_month):
            db_connection = connect_to_database()
            encaissement_query = """
            SELECT
                DATE(date_paye) AS jour,
                SUM(montant_pret) AS montant_pret,
                SUM(CASE 
                        WHEN mutualiseur = 1 AND date_paye > '2025-03-13' THEN investissement
                        WHEN mutualiseur <> 1 THEN investissement
                        ELSE 0 
                    END
                    ) AS investissement,
                0 AS prix_achat,
                0 AS investissement_2,
                0 AS depot_argent
            FROM contract
            WHERE ajustement <> 1 AND mutualiseur <> 0
            GROUP BY DATE(date_paye)

            UNION ALL

            SELECT
                DATE(reg_date) AS jour,
                0 AS montant_pret,
                0 AS investissement,
                SUM(prix_achat) AS prix_achat,
                SUM(investissement) AS investissement_2,
                0 AS depot_argent
            FROM bijou_achat
            WHERE mutualiseur <> 0
            GROUP BY DATE(reg_date)

            UNION ALL

            SELECT
                DATE(date_transaction) AS jour,
                0 AS montant_pret,
                0 AS investissement,
                0 AS prix_achat,
                0 AS investissement_2,
                SUM(montant) AS depot_argent
            FROM transactions_m
            WHERE type = 'depot'
            GROUP BY DATE(date_transaction);
            """  
            df_encaissement = pd.read_sql(encaissement_query, db_connection)
            db_connection.close()

            # Charger les décaissements
            db_connection = connect_to_database()
            decaissement_query = """
            SELECT
                DATE(reg_date) AS jour,
                SUM(montant_pret) AS montant_pret,
                0 AS investissement,
                0 AS prix_achat,
                0 AS retrait_argent
            FROM contract
            WHERE ajustement <> 1 AND mutualiseur <> 0
            GROUP BY DATE(reg_date)

            UNION ALL

            SELECT
                DATE(date_paye) AS jour,
                0 AS montant_pret,
                SUM(investissement) AS investissement,
                0 AS prix_achat,
                0 AS retrait_argent
            FROM contract
            WHERE ajustement = 1
            GROUP BY DATE(date_paye)

            UNION ALL

            SELECT
                DATE(reg_date) AS jour,
                0 AS montant_pret,
                0 AS investissement,
                SUM(prix_achat) AS prix_achat,
                0 AS retrait_argent
            FROM bijou_achat
            WHERE mutualiseur <> 0
            GROUP BY DATE(reg_date)

            UNION ALL

            SELECT 
                DATE(date_transaction) AS jour,
                0 AS montant_pret,
                0 AS investissement,
                0 AS prix_achat,
                SUM(montant) AS retrait_argent
            FROM transactions_m
            WHERE type = 'retrait'
            GROUP BY DATE(date_transaction);

            """  
            df_decaissement = pd.read_sql(decaissement_query, db_connection)
            db_connection.close()

            df_encaissement = df_encaissement.groupby("jour").sum().reset_index()
            df_encaissement["encaissement_total"] = (
                df_encaissement["montant_pret"]
                + df_encaissement["investissement"]
                + df_encaissement["prix_achat"]
                + df_encaissement["investissement_2"]
                + df_encaissement["depot_argent"]
            )
            df_decaissement = df_decaissement.groupby("jour").sum().reset_index()
            df_decaissement["decaissement_total"] = (
                df_decaissement["montant_pret"]
                + df_decaissement["investissement"]
                + df_decaissement["prix_achat"]
                + df_decaissement["retrait_argent"]
            )

            df_flux = pd.merge(df_encaissement[["jour", "encaissement_total"]], 
                               df_decaissement[["jour", "decaissement_total"]], on="jour",         
                               how="outer").fillna(0)

            df_flux["jour"] = pd.to_datetime(df_flux["jour"])
            # Fusion
            df_flux["trésorerie_nette"] = df_flux["encaissement_total"] - df_flux["decaissement_total"]
            df_flux["trésorerie_cumulée"] = df_flux["trésorerie_nette"].cumsum()

            df_flux_filtered = df_flux[
                (df_flux['jour'].dt.year == selected_year) &
                (df_flux['jour'].dt.month == selected_month)
            ]

            df_flux_filtered['trésorerie_nette'] = df_flux_filtered['encaissement_total'] - df_flux_filtered['decaissement_total']
            df_flux_filtered['trésorerie_cumulée'] = df_flux_filtered['trésorerie_nette'].cumsum()

            return df_flux_filtered
    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox(
            "Sélectionner l'année", 
             list(range(2022, now().year + 1))[::-1],
             key = 'invest_annee'
        )

    mois_fr = [
        "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]

    current_month = now().month

    with col2:
        selected_month = st.selectbox(
            "Sélectionner le mois",
             options=list(range(1, 13)),  
             index=current_month - 1,  
             format_func=lambda x: mois_fr[x - 1],
             key = 'invest_mois'
        )

    df_flux_filtered = get_flux_data(selected_year, selected_month)
    df_flux_filtered["encaissement_total"] = df_flux_filtered["encaissement_total"].round(0).astype(int)
    df_flux_filtered["decaissement_total"] = df_flux_filtered["decaissement_total"].round(0).astype(int)
    df_flux_filtered["trésorerie_nette"] = df_flux_filtered["trésorerie_nette"].round(0).astype(int)
    df_flux_filtered["trésorerie_cumulée"] = df_flux_filtered["trésorerie_cumulée"].round(0).astype(int)
    df_affichage = df_flux_filtered[['jour', 'encaissement_total', 'decaissement_total', 'trésorerie_nette', 'trésorerie_cumulée']]
    df_affichage.columns = ['Jour', 'Encaissement', 'Décaissement', 'Trésorerie Nette', 'Trésorerie Cumulée']
    st.dataframe(
        df_affichage.style.format({
            "Encaissement": lambda x: f"{x:,}".replace(",", " ") + " FCFA", 
            "Décaissement": lambda x: f"{x:,}".replace(",", " ") + " FCFA",
            "Trésorerie Nette": lambda x: f"{x:,}".replace(",", " ") + " FCFA",
            "Trésorerie Cumulée": lambda x: f"{x:,}".replace(",", " ") + " FCFA"
            }), use_container_width=True, hide_index=True
    )
    st.subheader("Évolution des flux sur le mois sélectionné")
    df_flux_filtré = get_flux_data(selected_year, selected_month).sort_values("jour")

    line_chart = alt.Chart(df_flux_filtré).transform_fold(
        ["trésorerie_nette", "trésorerie_cumulée"],
        as_=["Type", "Montant"]
    ).mark_line(point=True).encode(
        x="jour:T",
        y=alt.Y("Montant:Q", title="Montant (FCFA)"),
        color=alt.Color("Type:N", title="Type de flux", scale=alt.Scale(domain=["trésorerie_nette", "trésorerie_cumulée"], range=["#2ecc71", "#e74c3c"]))
    )

    seuil = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[5,5],  
        color='red'
    ).encode(
        y='y:Q'
    ).properties(
        title="Évolution des flux sur le mois sélectionné"
    )

    chart = alt.layer(line_chart, seuil).properties(
        width="container",
        height=400
    ).interactive()

    if not df_flux_filtré.empty:
        dernier_jour = df_flux_filtré["jour"].max()
        ligne_derniere_jour = df_flux_filtré.loc[df_flux_filtré["jour"] == dernier_jour]

        if not ligne_derniere_jour.empty:
            dernier_valeur = ligne_derniere_jour["trésorerie_cumulée"].values[0]

            couleur = "red" if dernier_valeur > 0 else "green"
            icone = "⚠️" if dernier_valeur > 0 else "✅"

            col1, col2 = st.columns([6, 1])
            with col1:
                st.altair_chart(chart, use_container_width=True)
            with col2:
                st.markdown(f"""
                    <style>
                    .alerte-flux {{
                        background-color: #fefefe;
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 6px solid {couleur};
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                        animation: fadeIn 1s ease-in;

                }}    
                @keyframes fadeIn {{
                        0% {{ opacity: 0; transform: translateY(10px); }}
                        100% {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>

                <div class="alerte-flux">
                    <h6 style="margin: 0;">{icone} Attente d'affectation </h6>
                <p style="font-size: 14px; color: {couleur}; margin: 0;"><strong>{dernier_valeur:,.0f} FCFA</strong></p>
                </div>
        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Aucune donnée de trésorerie disponible pour le dernier jour.")
    else: 
        st.warning("⚠️ Aucune donnée disponible pour le mois sélectionné.")

elif page == "Fidelor":
    st.title("💸 Fidelor")

    def get_chiffre_affaire_aor(month, year):
        db_connection = connect_to_database()
        query = """
        SELECT SUM(CASE WHEN paiement <> 'non payé' THEN fidelor_frais ELSE 0 END)  AS chiffre_affaire_aor
        FROM contract
        WHERE MONTH(date_paye) = %s AND YEAR(date_paye) = %s AND ajustement <> 1
        """
        with db_connection.cursor() as cursor:
            cursor.execute(query, (month, year))
            result = cursor.fetchone()
        return result[0] or 0
        db_connection.close()
        
    def get_chiffre_affaire_aoi(month, year):
        db_connection = connect_to_database()
        query = """
        SELECT SUM(CASE WHEN prix_valoriser <> 0 THEN fidelor ELSE 0 END) AS chiffre_affaire_aoi
        FROM bijou_achat
        WHERE MONTH(reg_date) = %s AND YEAR(reg_date) = %s
        """
        with db_connection.cursor() as cursor:
            cursor.execute(query, (month, year))
            result = cursor.fetchone()
        return result[0] or 0
        db_connection.close()
        
    st.subheader(" Les indicateurs financiers pour Fidelor")     
    col1, col2 = st.columns(2)    
    with col1:
        selected_year = st.selectbox(
           "Sélectionner l'année", 
            list(range(2022, now().year + 1))[::-1],
            key = 'fidelor_annee'
        )

    mois_fr = [
         "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
         "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
    ]

    current_month = now().month

    with col2:
        selected_month = st.selectbox(
            "Sélectionner le mois",
            options=list(range(1, 13)),
            index=current_month - 1,
            format_func=lambda x: mois_fr[x - 1],
            key = 'fidelor_mois'
        )

    chiffre_affaire_aor = get_chiffre_affaire_aor(selected_month, selected_year)
    chiffre_affaire_aoi = get_chiffre_affaire_aoi(selected_month, selected_year)

    col1, col2 = st.columns(2)

    chiffre_affaire_aor_fcfa = f"{round(chiffre_affaire_aor):,}".replace(",", " ") + " FCFA"
    chiffre_affaire_aoi_fcfa = f"{round(chiffre_affaire_aoi):,}".replace(",", " ") + " FCFA"
    with col1:
        display_card("Plus-values sur AOR", chiffre_affaire_aor_fcfa, "💼")

    with col2:
        display_card("Plus-values sur AOI", chiffre_affaire_aoi_fcfa, "💍")


    data_pie = pd.DataFrame({
        "Activité": ["Achat réméré", "Achat immédiat"],
        "Plus-values": [chiffre_affaire_aor, chiffre_affaire_aoi]
    })

    fig_pie = px.pie(
        data_pie,
        names="Activité",
        values="Plus-values",
        color="Activité",
        color_discrete_map={"Achat réméré": "#ADD8E6", "Achat immédiat": "#8A2BE2"},  
        hole=0.4 
    )

    fig_pie.update_traces(textinfo='percent+label')

    chiffre_affaire_total = chiffre_affaire_aor + chiffre_affaire_aoi

    seuil = 8000000  
    pourcentage = (chiffre_affaire_total / seuil) * 100 if seuil > 0 else 0

    if pourcentage < 50:
        statut = "Seuil critique"
        couleur_zone = "red"
    elif 50 <= pourcentage < 100:
        statut = "Vers l'objectif"
        couleur_zone = "orange"
    else:
        statut = "Objectif atteint"
        couleur_zone = "green"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=chiffre_affaire_total,
        number={'font': {'size': 40}}, 
        delta={"reference": seuil, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        gauge={
            'axis': {'range': [0, seuil * 1.5],
                     'tickformat': ',.0f'    
            },
            'bar': {'color': couleur_zone},
            'steps': [
                {'range': [0, seuil * 0.5], 'color': 'lightcoral'},
                {'range': [seuil * 0.5, seuil], 'color': 'moccasin'},
                {'range': [seuil, seuil * 1.5], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': seuil
            }
         },
         title={'text': f"<b>{statut}</b>"}
    ))

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.plotly_chart(fig_gauge, use_container_width=True)


    st.subheader("🧾 Tableau de comptabilité pour Fidelor")
    years = list(range(2022, now().year + 1))
    current_year = now().year

    selected_year = st.selectbox(
       "Sélectionner l'année pour le tableau de comptabilité",
        years,
        index=years.index(current_year)
    )


    db_connection = connect_to_database()
    query_aor = """
    SELECT MONTH(date_paye) AS mois, SUM(fidelor_frais) AS ca_aor, SUM(CASE WHEN mutualiseur = 0 AND promotteur = 0 THEN montant_pret ELSE 0 END) AS ca_aor_2
    FROM contract
    WHERE YEAR(date_paye) = %s AND paiement <> 'non payé' AND ajustement <> 1
    GROUP BY mois
    """
    df_aor = pd.read_sql(query_aor, db_connection, params=(selected_year,))
    db_connection.close()

    db_connection = connect_to_database()
    query_aoi = """
    SELECT MONTH(reg_date) AS mois, SUM(CASE WHEN prix_valoriser <> 0 THEN fidelor ELSE 0 END) AS ca_aoi,  SUM(CASE WHEN mutualiseur = 0 AND promotteur = 0 THEN prix_achat ELSE 0 END) as ca_aoi_2
    FROM bijou_achat
    WHERE YEAR(reg_date) = %s
    GROUP BY mois
    """
    df_aoi = pd.read_sql(query_aoi, db_connection, params=(selected_year,))
    db_connection.close()

    df_encaissements = pd.merge(df_aor, df_aoi, on="mois", how="outer").fillna(0)
    dataset_1 = df_encaissements
    df_encaissements["Encaissements"] = df_encaissements["ca_aor"] + df_encaissements["ca_aor_2"] + df_encaissements["ca_aoi"] + df_encaissements["ca_aoi_2"]

    db_connection = connect_to_database()
    query_tva_aor = """
    SELECT MONTH(date_paye) AS mois, SUM(fidelor_frais) * 0.09 AS tva_aor
    FROM contract
    WHERE YEAR(date_paye) = %s AND paiement <> 'non payé' AND ajustement <> 1
    GROUP BY mois
    """
    df_tva_aor = pd.read_sql(query_tva_aor, db_connection, params=(selected_year,))
    db_connection.close()

    db_connection = connect_to_database()
    query_tva_aoi = """
    SELECT MONTH(reg_date) AS mois, SUM(CASE WHEN prix_valoriser <> 0 THEN fidelor ELSE 0 END) * 0.09 AS tva_aoi
    FROM bijou_achat
    WHERE YEAR(reg_date) = %s
    GROUP BY mois
    """
    df_tva_aoi = pd.read_sql(query_tva_aoi, db_connection, params=(selected_year,))
    db_connection.close()

    db_connection = connect_to_database()
    query_prets = """
    SELECT MONTH(reg_date) AS mois, SUM(montant_pret) AS prets
    FROM contract
    WHERE YEAR(reg_date) = %s AND mutualiseur = 0 AND promotteur = 0
    GROUP BY mois
    """
    df_prets = pd.read_sql(query_prets, db_connection, params=(selected_year,))
    db_connection.close()
    db_connection = connect_to_database()
    query_achats = """
    SELECT MONTH(reg_date) AS mois, SUM(prix_achat) AS achats
    FROM bijou_achat
    WHERE YEAR(reg_date) = %s AND mutualiseur = 0 AND promotteur = 0
    GROUP BY mois
    """
    df_achats = pd.read_sql(query_achats, db_connection, params=(selected_year,))
    db_connection.close()

    db_connection = connect_to_database()
    query_invest = """
    SELECT MONTH(date_paye) AS mois, SUM(investissement) AS investissements
    FROM contract
    WHERE YEAR(date_paye) = %s AND ajustement = 1 AND paiement <> 'non payé'
    GROUP BY mois
    """
    df_invest = pd.read_sql(query_invest, db_connection, params=(selected_year,))
    db_connection.close()
    mois_actuel = now().month
    annee_actuelle = now().year
    if selected_year == annee_actuelle:
        df_decaissements = pd.DataFrame({'mois': range(1, mois_actuel + 1)})
    else:
        df_decaissements = pd.DataFrame({'mois': range(1, 13)})
    
    df_decaissements = df_decaissements.merge(df_tva_aor,  on="mois", how="left")
    df_decaissements = df_decaissements.merge(df_tva_aoi, on="mois", how="left")
    df_decaissements = df_decaissements.merge(df_prets, on="mois", how="left")
    df_decaissements = df_decaissements.merge(df_achats, on="mois", how="left")
    df_decaissements = df_decaissements.merge(df_invest, on="mois", how="left")

    df_decaissements.fillna(0, inplace=True)
    df_charges = pd.read_excel("charges.xlsx")
    df_charges = df_charges[df_charges["année"] == selected_year]
    mois_map = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
    }
    df_charges["mois_num"] = df_charges["mois"].str.lower().map(mois_map)
    df_charges["charges_totales"] = (
        df_charges["salaires"] +
        df_charges["électricité"] +
        df_charges["loyer"] +
        df_charges["matériels"] +
        df_charges["autres"]
    )
    charges_par_mois = df_charges.set_index("mois_num")["charges_totales"].to_dict()
    

    df_decaissements["charges_salaire"] = df_decaissements["mois"].map(charges_par_mois)
    df_decaissements["charges_salaire"].fillna(0, inplace=True)
    dataset_2 = df_decaissements
    df_decaissements["Décaissements"] = (
        df_decaissements["tva_aor"] +
        df_decaissements["tva_aoi"] +
        df_decaissements["prets"] +
        df_decaissements["achats"] +
        df_decaissements["investissements"] +
        df_decaissements["charges_salaire"]
    )

    df_final = pd.merge(df_encaissements, df_decaissements, on="mois", how="outer")
    df_final.fillna(0, inplace=True) 
    df_final["Mois"] = df_final["mois"].apply(lambda x: mois_fr[x - 1])

    df_final["Bénéfice net"] = df_final["Encaissements"] - df_final["Décaissements"]

    def type_balance(row):
        if row["Bénéfice net"] > 0:
            return "✅ Excédent"
        elif row["Bénéfice net"] == 0:
            return "➖ Équilibré"
        else:
            return "❌ Déficit"
    df_final["Balance"] = df_final.apply(type_balance, axis=1)

    df_final = df_final.sort_values("mois")[["Mois", "Encaissements", "Décaissements", "Bénéfice net", "Balance"]].reset_index(drop=True)
    
    st.dataframe(
        df_final[["Mois", "Encaissements", "Décaissements", "Bénéfice net", "Balance"]]
        .rename(columns={
            "Encaissements": "Encaissements (FCFA)",
            "Décaissements": "Décaissements (FCFA)",
            "Bénéfice net": "Bénéfice net (FCFA)",
            "Balance": "Balance (FCFA)"
        })
        .style.format({
            "Encaissements (FCFA)": lambda x:  f"{x:,.0f}".replace(",", " "),
            "Décaissements (FCFA)": lambda x:  f"{x:,.0f}".replace(",", " "),
            "Bénéfice net (FCFA)": lambda x:  f"{x:,.0f}".replace(",", " ")
        }), use_container_width=True, hide_index=True
    )
    dataset_1["Mois"] = dataset_1["mois"].apply(lambda x: mois_fr[x - 1])
    dataset_2["Mois"] = dataset_2["mois"].apply(lambda x: mois_fr[x - 1])
    dataset_2 = dataset_2.sort_values("mois")[["Mois", "tva_aor", "tva_aoi", "prets", "achats", "investissements", "charges_salaire",
                                               "Décaissements"]].reset_index(drop=True)
    
    dataset_1 = dataset_1.sort_values("mois")[["Mois", "ca_aor",  "ca_aor_2", "ca_aoi", "ca_aoi_2", "Encaissements"]].reset_index(drop=True)

    st.subheader("Détails sur les décaissements d'argent")
    st.dataframe(
        dataset_2[["Mois", "tva_aor", "tva_aoi", "prets", "achats", "investissements", "charges_salaire",  "Décaissements"]]
        .rename(columns={
            "tva_aor": "TVA\AOR (FCFA)",
            "tva_aoi": "TVA\AOI (FCFA)",
            "prets": "Montant achat non mutualisé\AOR (FCFA)",
            "achats": "Montant achat non mutualisé\AOI (FCFA)",
            "investissements":"Gains versés sur contrats d'ajustement",
            "charges_salaire":"Charges courantes mensuelles (FCFA)",
            "Décaissements": "Total des décaissements"
        })
        .style.format({
            "TVA\AOR (FCFA)": lambda x:  f"{x:,.0f}".replace(",", " "),
            "TVA\AOI (FCFA)": lambda x:  f"{x:,.0f}".replace(",", " "),
            "Montant achat non mutualisé\AOR (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Montant achat non mutualisé\AOI (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Gains versés sur contrats d'ajustement": lambda x: f"{x:,.0f}".replace(",", " "),
            "Charges courantes mensuelles (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Total des décaissements": lambda x: f"{x:,.0f}".replace(",", " "),
            
        }), use_container_width=True, hide_index=True
    )

    st.subheader("Détails sur les encaissements d'argent")
    st.dataframe(
        dataset_1[["Mois", "ca_aor", "ca_aor_2", "ca_aoi", "ca_aoi_2", "Encaissements"]]
        .rename(columns={
            "ca_aor": "Gains fidelor\AOR (FCFA)",
            "ca_aor_2": "Montant achat contrat non mutualisé\AOR (FCFA)",
            "ca_aoi": "Gains fidelor\AOI (FCFA)",
            "ca_aoi_2": "Montant achat contrat non mutualisé\AOI (FCFA)",
            "Encaissements": "Total des encaissements"
        })
        .style.format({
            "Gains fidelor\AOR (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Montant achat contrat non mutualisé\AOR (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Gains fidelor\AOI (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Montant achat contrat non mutualisé\AOI (FCFA)": lambda x: f"{x:,.0f}".replace(",", " "),
            "Total des encaissements": lambda x: f"{x:,.0f}".replace(",", " ") 
            
        }), use_container_width=True, hide_index=True
    )
    