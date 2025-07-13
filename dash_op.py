import dash
from dash import html, dcc, Input, Output, State
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# === Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Setup embeddings + vectorstore
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
vectorstore = Chroma(
    persist_directory="chroma_project_db",
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever()

# === Setup Gemini Flash LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# === Create Dash App
app = dash.Dash(__name__)
app.title = "SVM Explorer – Chatbot"

app.layout = html.Div([
    html.Div([
        html.H2("💬 Project Chatbot"),
        dcc.Input(
            id="user-input",
            type="text",
            placeholder="Ask something about the models...",
            style={"width": "100%", "padding": "8px"}
        ),
        html.Button("Ask", id="submit-btn", n_clicks=0),
        html.Div(id="chat-output", style={"marginTop": "20px", "whiteSpace": "pre-wrap"})
    ], style={
        "width": "30%",
        "padding": "20px",
        "borderRight": "1px solid #ccc",
        "height": "100vh",
        "position": "fixed",
        "left": 0,
        "top": 0,
        "overflowY": "auto",
        "backgroundColor": "#f9f9f9"
    }),

    html.Div([
        html.H1("🏡 California Housing – SVM Regression Explorer"),
        html.P("Main dashboard content goes here...")
    ], style={"marginLeft": "32%", "padding": "20px"})
])


@app.callback(
    Output("chat-output", "children"),
    Input("submit-btn", "n_clicks"),
    State("user-input", "value")
)
def update_chat(n_clicks, user_query):
    if n_clicks == 0 or not user_query:
        return ""
    response = qa_chain(user_query)
    return f"**Answer:**\n{response['result']}"


if __name__ == "__main__":
    app.run(debug=True)
