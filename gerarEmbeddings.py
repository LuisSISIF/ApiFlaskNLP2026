import os
from dotenv import load_dotenv
import google.generativeai as generativeai
import pandas as pd
import numpy as np
import pickle

# Carrega variáveis do .env
load_dotenv()

# Obtém a chave da API
chave_secreta = os.environ.get('GEMINI_API_KEY', 'AIzaSyDGMuhiSu_TNiYNrED5gnLIhbPheBR9noY')

if not chave_secreta:
    raise ValueError(
        "A variável de ambiente GEMINI_API_KEY não foi encontrada."
    )

# Configura Gemini
generativeai.configure(api_key=chave_secreta)

# URL da planilha publicada em CSV
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQP1f3uFCZ1RSM3uQ3kW76b3ThhPAv6yauoCdMdpwH3UNaDXpYI-Q7JSvuBl-3Org/pub?output=csv"

# Carrega dataset
df = pd.read_csv(csv_url)

print("Colunas encontradas:")
print(df.columns)

print("\nPrimeiros registros:")
print(df.head())

# Cria texto consolidado para embeddings
df["TextoEmbedding"] = df.apply(
    lambda row: (
        f"Categoria: {row['Categoria']}\n"
        f"Palavras-chave: {row['Palavras-chave']}\n"
        f"Pergunta: {row['Pergunta']}\n"
        f"Resposta: {row['Resposta']}"
    ),
    axis=1
)

# Modelo de embedding
MODEL = "models/gemini-embedding-001"


def gerar_embeddings(titulo, texto):
    resultado = generativeai.embed_content(
        model=MODEL,
        content=texto,
        task_type="retrieval_document",
        title=titulo
    )

    return resultado["embedding"]


def buscar_resposta(consulta, dataset):
    embedding_consulta = generativeai.embed_content(
        model=MODEL,
        content=consulta,
        task_type="retrieval_query"
    )

    produtos_escalares = np.dot(
        np.stack(dataset["Embeddings"]),
        embedding_consulta["embedding"]
    )

    indice = np.argmax(produtos_escalares)

    print(f"\nSimilaridade encontrada: {produtos_escalares[indice]:.4f}")

    return {
        "pergunta": dataset.iloc[indice]["Pergunta"],
        "resposta": dataset.iloc[indice]["Resposta"],
        "categoria": dataset.iloc[indice]["Categoria"]
    }


print("\nGerando embeddings...")
df["Embeddings"] = df.apply(
    lambda row: gerar_embeddings(
        row["Pergunta"],
        row["TextoEmbedding"]
    ),
    axis=1
)

print("\nEmbeddings gerados com sucesso!")

# Salva dataset processado
pickle.dump(
    df,
    open("datasetEmbeddings.pkl", "wb")
)

print("\nArquivo datasetEmbeddings.pkl salvo com sucesso.")
