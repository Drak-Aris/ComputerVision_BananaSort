import cv2
import numpy as np
import pandas as pd
import os

def extract_banana_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 256:
        scale = 256 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    lower_dark = np.array([0, 40, 0])
    upper_dark = np.array([179, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    mask_fruit_initial = cv2.bitwise_or(mask_yellow, mask_green)
    mask_fruit_initial = cv2.bitwise_or(mask_fruit_initial, mask_brown)
    mask_fruit_initial = cv2.bitwise_or(mask_fruit_initial, mask_dark)

    kernel = np.ones((5, 5), np.uint8)
    mask_fruit_initial = cv2.morphologyEx(mask_fruit_initial, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fruit_initial = cv2.morphologyEx(mask_fruit_initial, cv2.MORPH_CLOSE, kernel, iterations=2)


    contours_fruit, _ = cv2.findContours(mask_fruit_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_fruit:
        main_contour = max(contours_fruit, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(main_contour)
        largeur_banane = bw
        hauteur_banane = bh

        mask_fruit_filled = np.zeros_like(mask_fruit_initial)
        cv2.drawContours(mask_fruit_filled, [main_contour], -1, 255, -1)
    else:
        largeur_banane = w
        hauteur_banane = h
        mask_fruit_filled = np.ones((h, w), dtype=np.uint8) * 255


    total_fruit_pixels = cv2.countNonZero(mask_fruit_filled)
    if total_fruit_pixels == 0:
        total_fruit_pixels = 1

    mask_black_all = cv2.inRange(hsv, lower_dark, upper_dark)
    mask_black = cv2.bitwise_and(mask_black_all, mask_black_all, mask=mask_fruit_filled)
    pct_noir = (cv2.countNonZero(mask_black) / total_fruit_pixels) * 100
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    taches_noires = sum(1 for c in contours_black if cv2.contourArea(c) > 5)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 30, 255])
    mask_white_all = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(mask_white_all, mask_white_all, mask=mask_fruit_filled)
    kernel_small = np.ones((3, 3), np.uint8)
    mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_small, iterations=1)
    pct_blanc = (cv2.countNonZero(mask_white_clean) / total_fruit_pixels) * 100
    contours_white, _ = cv2.findContours(mask_white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    taches_blanches = sum(1 for c in contours_white if cv2.contourArea(c) > 10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_fruit = cv2.bitwise_and(edges, edges, mask=mask_fruit_filled)
    contours_fissures, _ = cv2.findContours(edges_fruit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    longueur_fissures = sum(cv2.arcLength(c, False) for c in contours_fissures)


    features = {
        "pct_noir": round(pct_noir, 2),
        "nb_taches_noires": taches_noires,
        "pct_blanc": round(pct_blanc, 2),
        "nb_taches_blanches": taches_blanches,
        "longueur_fissures": round(longueur_fissures, 2)
    }
    return features


def process_folder(root_folder, label):
    data = []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    print(f"Traitement du dossier '{root_folder}'...")
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_path = os.path.join(root, file)
                try:
                    feats = extract_banana_features(image_path)
                    feats["classe"] = label
                    data.append(feats)
                except Exception as e:
                    print(f"⚠️ Erreur sur {image_path} : {e}")
    return data


def build_dataset(lien_sain, lien_malade, output_file):
    data_sain = process_folder(lien_sain, "sain")
    data_malade = process_folder(lien_malade, "malade")

    all_data = data_sain + data_malade
    df = pd.DataFrame(all_data)

    if df.empty:
        print("Aucune donnée à sauvegarder.")
        return None

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"\n🎉 Dataset final sauvegardé : {output_file}")
    print(f"   → {len(df)} images au total, classes : {df['classe'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    path_sain = "../dataset/etat_sante/saint/training"
    path_malade = "../dataset/etat_sante/malade/training"

    output_parquet = "../dataset/features_bananes_etatsante.parquet"

    df = build_dataset(path_sain, path_malade, output_parquet)

    if df is not None:
        print("\n--- APERÇU DES DONNÉES ---")
        lire = pd.read_parquet(output_parquet)
        print(lire.head())