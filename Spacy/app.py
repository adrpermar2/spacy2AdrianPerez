# from crypt import methods
# from flask import Flask, render_template, request
# import spacy
# from spacy.tokens import Span


# app = Flask(__name__)



# @app.route("/", methods=["GET"])
# def home():
#     data = request.form.get("rawtext")
#     return render_template("index.html", data=data)
    
# @app.route("/process",methods=["POST"])
# def process():
#     nlp = spacy.load("es_core_news_sm")

#     doc = nlp(request.form.get("rawtext"))
#     for ent in doc.ents:
#         # Imprime en pantalla el texto y la URL de Wikipedia de la entidad
#         lista = ""
#         lista = lista + ent.text

        
#         lista = lista.split(" ")
        

#     return render_template("process.html", lista=lista)


# if __name__ == "__main__":
#     app.run(debug = True)
from flask import Flask, render_template, request
from crypt import methods
from unittest import result
import spacy
from spacy.tokens import Span
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from classifier import SentimentClassifier
from sklearn.externals import joblib
import joblib

app = Flask(__name__)



@app.route("/", methods=["GET"])
def home():
    data = request.form.get("rawtext")
    return render_template("index.html", data=data)
    
@app.route("/process",methods=["POST"])
def process():
    nlp = spacy.load("es_core_news_sm")

    doc = nlp(request.form.get("rawtext"))
    texto = str(doc)
    hola = "hola mi nombre es Adri"
    sid = SentimentIntensityAnalyzer()
    resultados = sid.polarity_scores(texto)

    clf = SentimentClassifier()
    sentimiento = clf.predict(hola)
    for ent in doc.ents:
        # Imprime en pantalla el texto y la URL de Wikipedia de la entidad
        lista = ""
        lista = lista + ent.text

        nombre = "Soy adrian"

        lista = lista.split(" ")

        tipo = ""
        tipo = tipo + ent.label_
        tipo = tipo.split(" ")
        return render_template("process.html", lista=lista, tipo=tipo, resultados=resultados, sentimiento=sentimiento)


if __name__ == "__main__":
    app.run(debug = True)