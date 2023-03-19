
import streamlit as st
import pandas as pd
from pycaret import *
from pycaret.classification import *

# Afficher le titre de l'application
st.title("Application de traitement de langage naturel")

# Charger les données
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Afficher les données
    st.subheader("Données")
    st.write(data)

    # Supprimer des colonnes
    cols_to_drop = st.multiselect("Choisir les colonnes à supprimer", data.columns)
    if cols_to_drop:
        data = data.drop(cols_to_drop, axis=1)
        st.subheader("Données après suppression de colonnes")
        st.write(data)

    # Ajouter des colonnes
    new_cols = st.text_input("Ajouter des colonnes (séparées par des virgules)")
    if new_cols:
        new_cols_list = [col.strip() for col in new_cols.split(",")]
        for col in new_cols_list:
            data[col] = ""
        st.subheader("Données après ajout de colonnes")
        st.write(data)

    # Définir le target et les données
    target_col = st.selectbox("Choisir la colonne cible", data.columns)
    setup_data = setup(data=data, target=target_col, ignore_features=[target_col])

    # Créer un modèle LDA
    lda_model = create_model("lda")

    # Résumer le modèle LDA
    st.subheader("Résumé du modèle LDA")
    summary = summarize_model(lda_model)
    st.write(summary)

    # Traduire le texte
    st.subheader("Traduction")
    text_to_translate = st.text_input("Entrez le texte à traduire")
    if text_to_translate:
        target_language = st.selectbox("Choisir la langue cible", ["fr", "en", "es", "de", "it"])
        translated_text = translate(text_to_translate, target_language)
        st.write(translated_text)

    # Reconnaissance d'entité
    st.subheader("Reconnaissance d'entité")
    text_for_ner = st.text_input("Entrez le texte pour la reconnaissance d'entité")
    if text_for_ner:
        ner_result = ner(text_for_ner)
        st.write(ner_result)

    # Représentation graphique de bigrammes et n-grammes
    st.subheader("Représentation graphique de bigrammes et n-grammes")
    text_for_bigrams = st.text_input("Entrez le texte pour la représentation graphique de bigrammes et n-grammes")
    if text_for_bigrams:
        num_of_words = st.slider("Nombre de mots", 2, 10, 2)
        plot_type = st.selectbox("Type de graphe", ["bigram", "trigram", "quadgram"])
        plot_model(lda_model, plot=plot_type, topic_num=0, num_words=num_of_words)

