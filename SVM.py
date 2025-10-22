import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import joblib
from streamlit_lottie import st_lottie
import requests
from langchain.chat_models import ChatOpenAI

# === Load environment ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Setup embeddings and vector DB
embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)
import faiss 

vectorstore = FAISS.load_local(
    "faiss_project_db",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# === Setup Gemini Flash LLM + QA chain
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-lite",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.2,
# )

llm = ChatOpenAI(
    model_name="gpt-4o-mini",         # GPT-4o includes the mini variant automatically
    temperature=0.2,
    openai_api_key=openai_api_key
)


# Create a simple QA chain using the modern LangChain approach
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question about the California Housing dataset and SVM models.
If you don't know the answer, say you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}
""")

# Create the QA chain
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


st.markdown("""
<style>
/* === General App Background === */
.stApp {
    background-color: #0e1117;
    color: #f1f1f1;
    font-family: 'Segoe UI', sans-serif;
}

/* === Headers and Titles === */
h1, h2, h3 {
    color: #0ff0fc;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* === Metric widgets === */
[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #00e1ff;
}

/* === Buttons === */
button[kind="primary"] {
    background-color: #0ff0fc !important;
    color: #0e1117 !important;
    font-weight: 600;
    border-radius: 10px;
}

/* === Sidebar styling === */
.css-1d391kg, .css-1lcbmhc {
    background-color: #1a1e29 !important;
    color: #f1f1f1 !important;
}

/* === Text input + sliders === */
input, textarea, .stSlider {
    background-color: #2b2e3b !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align: center; color: #0ff0fc; font-size: 48px;'>
    🏡 California Housing AI Dashboard
</h1>
<h4 style='text-align: center; color: #999;'>SVM & Random Forest Regression Explorer</h4>
""", unsafe_allow_html=True)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_ydo1amjm.json")
        st_lottie(animation, height=250)
    with col2:
        st.markdown("### 🤖 Powered by gpt-4o-mini + ChromaDB")
        st.markdown("Smart AI assistant with access to your latest model predictions and context.")

# === Sidebar Chatbot ===
with st.sidebar:
    st.markdown("## 💬 Project Chatbot")
    st.info("Ask about the models, predictions, or methodology. Or, if you're unsure, just tell the chat: 'Navigate me through the app.")

    user_input = st.text_input("Your question:")

    if user_input:
        use_context = any(kw in user_input.lower() for kw in ["prediction", "value", "median", "input", "features", "result", "model used"])

        context_lines = []

        if use_context:
            pred_val = st.session_state.get("last_prediction")
            model_name = st.session_state.get("prediction_model")
            features = st.session_state.get("input_features")

            if model_name:
                context_lines.append(f"The user selected the model **{model_name}** for prediction.")
            if pred_val:
                context_lines.append(f"The predicted median house value is **${pred_val * 100_000:,.0f}**.")
            if features:
                context_lines.append("The input features used for prediction were:")
                for key, value in features.items():
                    context_lines.append(f"- {key}: {value}")

        prediction_context = "\n".join(context_lines)
        full_query = prediction_context + "\n" + user_input if context_lines else user_input

        response = qa_chain.invoke(full_query)
        st.success(f"🧠 **Answer:** {response}")



# === Title and description ===
st.title("🏡 California Housing – SVM Regression Explorer")

st.markdown("""
Welcome to the **California Housing – SVM Regression Explorer**!

In this project, I focused on understanding and evaluating the performance of **Support Vector Machine (SVM)** models for regression, and how they compare to a **Random Forest Regressor**.

---

### 🎯 Project Objective

The goal was to explore how well different SVM-based regression models perform on the **California housing dataset** from `scikit-learn`, and to contrast their accuracy, training time, and behavior with a Random Forest model.
(The data contains information from the 1990 California census.)
            
### 🧪 What I Did

- Trained and evaluated:
  - `LinearSVR` — a fast linear model for regression.
  - `SVR` with RBF kernel — capable of modeling nonlinear patterns.
  - `Tuned SVR` — optimized using `RandomizedSearchCV` for better performance.
- Compared results to:
  - `RandomForestRegressor` — a robust tree-based ensemble model.

The evaluated using:
- **Root Mean Squared Error (RMSE)**
- **Training Time**
- (For tuned models) **Best hyperparameters found**

---

### 💬 Ask the Assistant

On the left side of the App, you’ll find a built-in **chatbot**. You can ask it anything about:
- The models used
- The dataset
- Performance comparisons
- SVM vs. tree-based models
- Even how the Chatbot is working 
- And much more
            
""")



st.markdown("## 🔍 The Tabs")
st.markdown("""
You have here 3 Tabs that you can check, each provides information and functionality of the project.
""")

@st.cache_resource
def load_models():
    import joblib
    models = {
        "LinearSVR": joblib.load("linear_svr.pkl"),
        "SVR (RBF)": joblib.load("svr.pkl"),
        "Tuned SVR": joblib.load("tuned_svr.pkl"),
        "Random Forest": joblib.load("random_forest.pkl"),
    }
    return models

models = load_models()


tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "⚙️ Model Comparison", "🔮 Prediction"])

with tab1:
    st.markdown("### 📄 Raw Dataset")
    df = fetch_california_housing(as_frame=True).frame
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.markdown("### 📈 Feature Distribution")
    feature = st.selectbox("Choose a feature", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

with tab2:
    st.markdown("### ⚙️ Compare Regression Models")
    model_choice = st.selectbox("Choose a model:", ["LinearSVR", "SVR (RBF)", "Tuned SVR", "Random Forest"])

    model_paths = {
        "LinearSVR": "linear_svr.pkl",
        "SVR (RBF)": "svr.pkl",
        "Tuned SVR": "tuned_svr.pkl",
        "Random Forest": "random_forest.pkl"
    }
    model = joblib.load(model_paths[model_choice])

    rmse_values = {
    "LinearSVR": "0.86",
    "SVR (RBF)": "0.58",
    "Tuned SVR": "0.58",
    "Random Forest": "0.49"
    }

    st.metric("📉 RMSE", rmse_values[model_choice])
    st.markdown(f"**Model:** `{model_choice}`")
    if model_choice == "Tuned SVR":
        st.code("""SVR(C=8.17, gamma=0.0846, epsilon=0.1508)""")
    elif model_choice == "Random Forest":
        st.code("""
        {'n_estimators': 185,
         'max_depth': None,
         'max_features': 'log2',
         'min_samples_split': 2,
         'min_samples_leaf': 3}
        """)

with tab3:
    st.markdown("### 🏠 Enter Features to Predict")
    st.markdown("After choosing a model and picking the input, you can ask chat about it!")
    # 1️⃣ Model selector (inside Prediction tab)
    pred_model_name = st.selectbox("Choose a model to use for prediction:", list(models.keys()), key="predict_model")

    # 2️⃣ Feature sliders
    med_inc = st.slider("Median Income", 0.5, 15.0, 5.0)
    ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    house_age = st.slider("House Age", 1, 50, 20)

    # 3️⃣ Prepare input
    X_full = df.drop(columns="MedHouseVal")
    feature_names = X_full.columns.tolist()
    input_data = X_full.mean().to_frame().T
    input_data["MedInc"] = med_inc
    input_data["AveRooms"] = ave_rooms
    input_data["HouseAge"] = house_age
    X_input = input_data[feature_names]

    # 4️⃣ Predict with selected model
    model = models[pred_model_name]
    y_pred = model.predict(X_input)[0]
    st.metric("💰 Predicted Median Value", f"${y_pred * 100_000:,.0f}")

    # 5️⃣ Store prediction for chatbot
    st.session_state["last_prediction"] = y_pred
    st.session_state["prediction_model"] = pred_model_name
    st.session_state["input_features"] = {
        "Median Income": med_inc,
        "Average Rooms": ave_rooms,
        "House Age": house_age
    }


