import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Charger le fichier CSV
file_path = 'atomic_data.csv'  # Remplacez par le chemin vers votre fichier CSV
atomic_data = pd.read_csv(file_path)

# Ajouter une colonne 'Total Revenue'
atomic_data['Total Revenue'] = atomic_data['Quantity'] * atomic_data['Unit Price']

# Convertir la colonne 'Transaction Date' en datetime
atomic_data['Transaction Date'] = pd.to_datetime(atomic_data['Transaction Date'])

# Grouper par mois et année et calculer le chiffre d'affaires mensuel
monthly_revenue = atomic_data.resample('M', on='Transaction Date')['Total Revenue'].sum()

# Créer un DataFrame avec la date et le chiffre d'affaires mensuel
data = monthly_revenue.reset_index()
data.columns = ['Date', 'Revenue']

# Extraire les caractéristiques temporelles
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Utiliser uniquement les années et les mois pour les prévisions
features = data[['Year', 'Month']]
target = data['Revenue']

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entraîner un modèle de Support Vector Regression
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Faire des prédictions pour l'évaluation
y_pred_svr = svr_model.predict(X_test)

# Évaluer le modèle
mse_svr = mean_squared_error(y_test, y_pred_svr)
st.write(f'Erreur quadratique moyenne (SVR): {mse_svr:.2f}')

# Interface utilisateur pour les prédictions dynamiques
st.sidebar.title("Prévision du chiffre d'affaires 🎨")
st.sidebar.markdown("<span style='color:white'>Prédisez le chiffre d'affaires futur à l'aide de SVM Regressor.</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

year = st.sidebar.number_input("Entrez l'année:", min_value=2022, max_value=2030, value=2024)
month = st.sidebar.number_input("Entrez le mois:", min_value=1, max_value=12, value=5)

# Bouton pour effectuer la prédiction
if st.sidebar.button('Prévoir'):
    # Préparer les données pour la prédiction
    prediction_input = pd.DataFrame({'Year': [year], 'Month': [month]})
    prediction = svr_model.predict(prediction_input)
    st.sidebar.markdown(f"<span style='color:white'>Prévision du chiffre d'affaires pour {year}-{month:02d}: {prediction[0]:.2f}</span>", unsafe_allow_html=True)

# Afficher les résultats et visualisations
st.sidebar.markdown("---")
st.sidebar.header("Résultats")
st.write(f"Chiffre d'affaires total: {data['Revenue'].sum():.2f}")
st.write("Chiffre d'affaires par produit:")
st.write(atomic_data.groupby('Product Name')['Total Revenue'].sum().sort_values(ascending=False).head())

st.write("Nombre de transactions par méthode de paiement:")
st.write(atomic_data['Payment Method'].value_counts())

st.write("Chiffre d'affaires par pays:")
st.write(atomic_data.groupby('Country')['Total Revenue'].sum().sort_values(ascending=False).head())

st.write('Tendance des ventes au fil du temps')
fig, ax = plt.subplots()
monthly_revenue.plot(kind='line', ax=ax, color='blue')
ax.set_title('Tendance des ventes au fil du temps', color='darkred')
ax.set_xlabel('Date', color='darkgreen')
ax.set_ylabel('Chiffre d\'affaires', color='darkgreen')
ax.tick_params(axis='x', colors='purple')
ax.tick_params(axis='y', colors='purple')
st.pyplot(fig)

# Custom styles
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f5;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)
