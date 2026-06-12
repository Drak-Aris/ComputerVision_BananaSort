from pathlib import Path
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

def training_exportmodel():
    df = pd.read_csv('../dataset/features_bananes_etatsante.csv')

    colonnes_features = [
        'pct_noir',
        'nb_taches_noires',
        'pct_blanc',
        'nb_taches_blanches',
        'longueur_fissures'
    ]

    x_train = df[colonnes_features]
    y_train = df['classe']

    modele_gnb = GaussianNB()
    modele_gnb.fit(x_train, y_train)

    fichier_modele = 'modele_classification_nb.pkl'
    joblib.dump(modele_gnb, fichier_modele)
    print(f"\nmodèle entrainé et sauvegardé avec succès sous : {fichier_modele}")


if __name__ == '__main__':
    """
    training_exportmodel()

    """
script = Path(__file__).parent
modele_save = script / 'modele_classification_nb.pkl'
modele_charge = joblib.load(modele_save)

chemin_test = script / '../dataset/features_bananes_etatsante_test.csv'
df_test = pd.read_csv(chemin_test)

colonnes_prises = [
    'pct_noir',
    'nb_taches_noires',
    'pct_blanc',
    'nb_taches_blanches',
    'longueur_fissures'
]

nouvelles_donnees = df_test[colonnes_prises]

predictions_test = modele_charge.predict(nouvelles_donnees)

for i, prediction in enumerate(predictions_test):
    print(f"Échantillon {i+1} : Prédit comme '{prediction}'")