import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

def training_exportmodel():
    df = pd.read_parquet('./dataset/features_bananes_etatsante.parquet')

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
    training_exportmodel()

    """
import pandas as pd
import joblib

modele_charge = joblib.load('modele_classification_nb.pkl')

nouvelles_donnees = pd.DataFrame({
    utiliser le fichier python features sur le dataset test pour generer les features et les coller directement dans dans cette variable nouvelles_donnees
})

predictions_test = modele_charge.predict(nouvelles_donnees)

for i, prediction in enumerate(predictions_nouvelles):
    print(f"Échantillon {i+1} : Prédit comme '{prediction}'")
    """