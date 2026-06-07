# 🌬️ AirFlow Inside

**Système intelligent de détection de conduite dangereuse par analyse des flux d'air**  
**Avec réseau d'alerte communautaire — Sans caméra, sans GPS**

---

## 💡 Concept

AirFlow Inside détecte les comportements dangereux à l’intérieur d’un véhicule (freinages brusques, virages serrés, conduite agressive) en analysant :

- Les flux d’air autour et dans le véhicule  
- La pression interne et les vibrations  
- La vitesse du vent et les conditions météo  

Le système **alerte le conducteur en temps réel** et peut partager les zones à risque entre véhicules (V2V), **sans GPS et sans caméra**.

---

## 📅 Pipeline Complet du Prototype (12 jours)

### 1️⃣ Matériel & Setup Initial (Jour 1-2)
- **Capteurs** : BMP280/BME680 (pression/température), anémomètre micro, capteur de pression interne  
- **Microcontrôleur** : ESP32 / Arduino Nano  
- **Logiciel** : Arduino IDE / PlatformIO pour hardware, Python pour traitement de données  
- **API météo** : OpenWeatherMap pour corriger les mesures en fonction du vent, température, pression

---

### 2️⃣ Collecte de Données (Jour 3-7)
- Montage du prototype dans le véhicule : capteurs avant, latéraux et internes  
- Mesures enregistrées : pression, température, vitesse du vent, vibrations, flux aérodynamique  
- Sessions de conduite variées : normal, agressif, freinage brusque  
- Labellisation manuelle des événements  
- Dataset final : capteurs + météo + label (CSV ou SQLite)

---

### 3️⃣ Pipeline IA (Machine Learning Classique, Jour 8-9)
- **Features utilisées** : `front_corr`, `sideL_corr`, `sideR_corr`, `d_front`, `d_sideL`, `d_sideR`, `asym`, `turbulence`, `internal_pressure`, `vibration`, `wind_speed`  
- **Modèle** : Logistic Regression (multi-class)  
- **Prétraitement** : Normalisation (`StandardScaler`), split train/test  
- **Évaluation** : Accuracy, inspection des coefficients  
- **Sauvegarde** : `model.pkl` et `scaler.pkl` pour utilisation en temps réel  

---

### 4️⃣ Interface & Démonstration (Jour 10-11)
- Dashboard simple (web ou mobile) pour visualiser :
  - État de conduite détecté  
  - Alertes en temps réel (LED / écran)  
  - Données météo actuelles  
- Historique des événements

---

### 5️⃣ Tests & Présentation (Jour 12)
- Validation en conditions réelles  
- Ajustement seuils et calibrage  
- Vidéo démo et slides explicatifs

---

## 💰 Budget Prototype (~12 jours)

| Composant                       | Coût estimé |
|---------------------------------|------------|
| Capteurs (BMP280, vent, pression interne) | ~30 dt |
| Microcontrôleur (ESP32/Arduino) | ~15 dt |
| Composants électroniques & câbles | ~20 dt |
| Boîtier + fixations              | ~10 dt |
| Divers (SD card, alimentation...) | ~10 dt |
| **Total**                        | **~85 dt** |

---

## ⚙️ Structure du Repo

- `airflow.py` — pipeline principal
- `ai_inference.py` — inférence temps réel
- `cv_train.py` — entraînement du modèle
- `sensor_gen.py` — génération de données capteurs


