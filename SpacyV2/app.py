from urllib.request import Request
from flask import Flask, render_template, request
from crypt import methods
from unittest import result
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
#from classifier import SentimentClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.cli.download import download
download(model="es_core_news_md")

import nltk
nltk.download('vader_lexicon')
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def get_lang_detector(nlp, name):
    return LanguageDetector()

def obtenerEntidades(texto):
    nlp = spacy.load("es_core_news_md")

    # Se pasa la funcion que nos permitira saber que idioma detecta
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)

    doc = nlp(texto)

    # Si el texto esta en ingles se vuelve a procesar el texto con el modelo en ingles
    if doc._.language['language'] == 'en':
        nlp = spacy.load("en_core_web_md")
        doc = nlp(texto)

    # Obtenemos todas las entidades
    entidades = [(ent.label_, ent.text) for ent in doc.ents]

    return entidades, doc._.language['language']


def obtenerSentimiento(language, texto):
    sid = SentimentIntensityAnalyzer()
    sentimiento = sid.polarity_scores(texto)
    if language == 'en':
        sid = SentimentIntensityAnalyzer()
        sentimiento = sid.polarity_scores(texto)
    # else:
    #     sid = SentimentClassifier()
    #     sentimiento = sid.predict(texto)

    return sentimiento


@app.route("/process", methods = ['POST'])
def procces_text():
    if request.method == 'POST':
        opciones = request.form['taskoption']
        entidades, language = obtenerEntidades(request.form['rawtext'])
        
        if opciones == "organization":
            entidad = "ORG"
        elif opciones == "location":
            entidad = "LOC"
        elif opciones == "person":
            if language == 'en':
                entidad = "PERSON"
            else:
                entidad = "PER"

        cantidad_resultados = [ent for ent in entidades if ent[0] == entidad]
    
        sentimiento = obtenerSentimiento(language, request.form['rawtext'])

    return render_template("index.html", num_of_results=len(cantidad_resultados), results=cantidad_resultados, sentimiento=sentimiento)

if __name__ ==  '__main__':
    app.run(debug = True)