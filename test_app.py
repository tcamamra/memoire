import pytest
from app import app

# Configuration du contexte de test pour l'application Flask
@pytest.fixture
def client():
    # Activation du mode test pour isoler les effets de bord
    app.config['TESTING'] = True
    # Création d'un client de test pour simuler les requêtes vers l'application
    with app.test_client() as client:
        yield client  # Retourne le client de test pour utilisation dans les fonctions de test

# Test de la route d'accueil pour s'assurer qu'elle est accessible et renvoie le contenu attendu
def test_home_route(client):
    # Envoi d'une requête GET vers la route d'accueil
    response = client.get('/')
    # Vérification que la réponse a un statut HTTP 200 (OK)
    assert response.status_code == 200
    # Vérification que le contenu de la réponse inclut un mot spécifique, ici "modèle"
    assert "modèle".encode('utf-8') in response.data

# Test de la route de prédiction avec un identifiant client valide
def test_predict_route_with_valid_id(client):
    # Simulation d'une requête POST avec un identifiant client valide
    response = client.post('/api/predict', data={'client_id': 123})
    # Vérification que la requête a réussi avec un statut HTTP 200
    assert response.status_code == 200
    # Extraction des données JSON de la réponse
    data = response.get_json()
    # Vérification que les données incluent une clé 'prediction'
    assert 'prediction' in data

# Test de la route de prédiction avec un identifiant client invalide
def test_predict_route_with_invalid_id(client):
    # Envoi d'une requête POST avec un identifiant client qui n'existe pas
    response = client.post('/api/predict', data={'client_id': 999})
    # Vérification que la requête retourne un statut HTTP 200 même en cas d'erreur
    assert response.status_code == 200
    # Récupération des données JSON de la réponse
    data = response.get_json()
    # Vérification que les données contiennent une clé 'error' et que 'prediction' est None
    assert 'error' in data
    assert data['prediction'] is None

# Point d'entrée pour exécuter les tests si le script est exécuté directement
if __name__ == '__main__':
    pytest.main()