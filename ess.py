from flask import Flask, jsonify, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Chargement des données clients à partir d'un fichier CSV pour l'analyse et la prédiction
df = pd.read_csv('df_dashboard.csv')

# Chargement du modèle de machine learning pré-entraîné pour les prédictions de scoring de crédit
with open('model_streamlit.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Page d'accueil de l'application, affichant un formulaire pour entrer l'ID client
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        client_id = int(request.form['client_id'])
        if client_id in df['id'].values:
            # Extraction des caractéristiques du client sans inclure l'ID
            client_features = df[df['id'] == client_id].drop('id', axis=1).iloc[0].values
            prediction = model.predict([client_features])[0]
            return render_template('result.html', prediction=prediction)
        else:
            return render_template('result.html', error="Identifiant client non trouvé dans nos enregistrements.")

    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        client_id = int(request.form['client_id'])
        if client_id in df['id'].values:
            # Préparation des caractéristiques du client pour la prédiction sans l'ID
            client_features = df[df['id'] == client_id].drop('id', axis=1).iloc[0].values
            prediction = model.predict([client_features])[0]
            result = {'prediction': int(prediction)}
        else:
            result = {'error': "ID client non reconnu dans nos données.", 'prediction': None}

        return jsonify(result)
    else:
        return jsonify({'error': "Méthode non supportée. Utilisez POST pour les prédictions."})

if __name__ == '__main__':
    app.run(port=8000)

