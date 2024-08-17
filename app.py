from flask import Flask, jsonify, render_template, request
import pandas as pd
import pickle
import lightgbm as lgb

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

# Route de test pour vérifier l'installation de LightGBM
@app.route('/test_lightgbm')
def test_lightgbm():
    try:
        import lightgbm as lgb
        return "LightGBM is installed correctly!"
    except ImportError as e:
        return f"Error: {str(e)}"

# Route de test pour une prédiction simple avec LightGBM
@app.route('/test_prediction')
def test_prediction():
    import lightgbm as lgb
    import numpy as np

    try:
        # Créez un modèle fictif pour le test
        data = np.array([[1, 2, 3], [4, 5, 6]])
        label = np.array([1, 0])
        train_data = lgb.Dataset(data, label=label)
        params = {'objective': 'binary'}

        # Entraînez un modèle LightGBM
        model = lgb.train(params, train_data, 2)

        # Faire une prédiction simple
        prediction = model.predict(np.array([[1, 2, 3]]))
        return f"Prediction: {prediction[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(port=8000)
