
import google.generativeai as generativeai
from google import genai
from google.genai import types
import numpy as np
import os
from dotenv import load_dotenv

model = 'models/gemini-embedding-001'
def gerarBuscarConsulta(consulta,dataset):
    embedding_consulta = generativeai.embed_content(model=model,
                                content=consulta,
                                task_type="retrieval_query",
                                )
    produtos_escalares = np.dot(np.stack(dataset["Embeddings"]), embedding_consulta['embedding']) # Calculo de distancia entre consulta e a base
    #print(embedding_consulta)
    #print(produtos_escalares)
    indice = np.argmax(produtos_escalares)
    #print(produtos_escalares[indice])
    if 'Resposta' in dataset.columns:
        return dataset.iloc[indice]['Resposta']
    elif 'Conteúdo' in dataset.columns:
        return dataset.iloc[indice]['Conteúdo']
    else:
        return dataset.iloc[indice].iloc[1]



modelo = 'gemini-2.5-flash'

def melhorarResposta(inputText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = modelo
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=inputText),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
      types.Part.from_text(text="""
      Você é um assistente baseado em RAG (Retrieval-Augmented Generation).
      Utilize exclusivamente o conteúdo recuperado da base de conhecimento para responder à consulta do usuário.
      Considere a consulta e o contexto recuperado, e gere uma resposta clara, objetiva e coerente, reescrevendo as informações de forma natural sem copiar literalmente o texto original.
      Não invente informações que não estejam presentes no contexto fornecido e não apresente múltiplas opções de resposta.
      """),
      ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text

