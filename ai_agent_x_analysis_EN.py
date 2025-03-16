import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Define the agent's state
class State(TypedDict):
    text: str  # The original tweet
    classification: str  # Tweet classification (compliment, complaint, technical support, question, etc.)
    entities: List[str]  # Extracted entities (e.g., products, problems, competitors)
    summary: str  # Short summary of the tweet
    sentiment: str  # Sentiment analysis (positive, neutral, negative)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tweet Classification Node
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following tweet into one of the categories: Compliment, Complaint, Technical Support, Question, or Other.\n\nTweet: {text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

# Entity Extraction Node
def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract key entities (Products, Services, Issues, Competitors) from the following tweet. Provide them as a comma-separated list.\n\nTweet: {text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

# Summarization Node
def summarize_node(state: State):
    summarization_prompt = PromptTemplate.from_template(
        """Summarize the following tweet in one short sentence.\n\nTweet: {input}\n\nSummary:"""
    )
    chain = summarization_prompt | llm
    response = chain.invoke({"input": state["text"]})
    return {"summary": response.content}

# Sentiment Analysis Node
def sentiment_analysis_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Determine the sentiment of the following tweet as Positive, Neutral, or Negative.\n\nTweet: {text}\n\nSentiment:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}

# Create the agent's workflow
workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)

workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", END)

# Compile the agent
app = workflow.compile()

# Example usage
tweet_sample = "your-sample-text-here"
state_input = {"text": tweet_sample}
result = app.invoke(state_input)

print("Classification:", result["classification"])
print("Entities:", result["entities"])
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
