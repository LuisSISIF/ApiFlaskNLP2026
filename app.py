from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import google.generativeai as generativeai
from google import genai
from google.genai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

from geminiFunctions import gerarBuscarConsulta, melhorarResposta

app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire application
modelo = 'gemini-2.5-flash'
modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl','rb'))
chave_secreta = os.getenv('GEMINI_API_KEY')
generativeai.configure(api_key=chave_secreta)


@app.route("/")
def home():
    consulta = "Quem é você ?"
    resposta = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resposta}"
    response = melhorarResposta(prompt)
    return response


@app.route("/api", methods=["POST"])
def results():
    # Verifique a chave de autorização
    auth_key = request.headers.get("Authorization")
    if auth_key and auth_key.startswith("Bearer "):
        auth_key = auth_key[len("Bearer "):]
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Requisição inválida. Envie um JSON válido."}), 400
        
    consulta = data.get("consulta")
    if not consulta:
        return jsonify({"error": "Parâmetro 'consulta' é obrigatório"}), 400
        
    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resultado}"
    response = melhorarResposta(prompt)
    return jsonify({"mensagem":  response})



