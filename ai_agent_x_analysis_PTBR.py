import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar o modelo OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Definição do estado do agente
class State(TypedDict):
    text: str  # Texto da postagem no X
    classification: str  # Classificação da postagem
    entities: List[str]  # Entidades extraídas
    summary: str  # Resumo da postagem
    sentiment: str  # Sentimento da postagem

# Classificação do tipo de postagem
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Classifique a seguinte postagem no X sobre a Vivo em uma das categorias:
        - Reclamação
        - Elogio
        - Suporte Técnico
        - Promoção
        - Outro
        
        Postagem: {text}
        
        Categoria:
        """
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

# Extração de entidades relevantes
def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extraia nomes de produtos, serviços, planos e problemas técnicos mencionados na postagem.
        Separe os itens por vírgula.
        
        Postagem: {text}
        
        Entidades:
        """
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

# Resumo da postagem
def summarize_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Resuma a seguinte postagem em uma frase curta e objetiva.
        
        Postagem: {text}
        
        Resumo:
        """
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

# Classificação de sentimento
def sentiment_analysis_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Determine o sentimento da seguinte postagem sobre a Vivo:
        - Positivo
        - Neutro
        - Negativo
        
        Postagem: {text}
        
        Sentimento:
        """
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}

# Criar fluxo do agente
workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)

# Definir sequência das operações
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", END)

# Compilar o fluxo
twitter_agent = workflow.compile()

# Testar o agente com uma postagem fictícia
sample_post = {
    "text": "aqui-sua-amostra-de-texto"
}
result = twitter_agent.invoke(sample_post)

# Exibir os resultados
print("Classificação:", result["classification"])
print("Entidades:", result["entities"])
print("Resumo:", result["summary"])
print("Sentimento:", result["sentiment"])
