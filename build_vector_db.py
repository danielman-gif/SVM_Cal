from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import os

# === Load API key from .env ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# === Your explanatory text for the chatbot ===
full_text = """
This project, titled California Housing – SVM Regression Explorer, focuses on evaluating
and comparing the performance of different regression models, particularly variants of
Support Vector Machines (SVM), in predicting median house values in California. The
dataset used for this analysis is the California housing dataset from the 1990 census,
which is available via scikit-learn. It contains various features such as median income,
average number of rooms, house age, population, and others, with the goal of predicting
the median house value. The full dataset includes over 16,000 rows, and for evaluation
purposes, a standard train-test split was applied, using 80% of the data for training and
20% for testing. Additionally, cross-validation with 5 folds was used during model
evaluation to ensure robustness and reduce the risk of overfitting.
The main objective of this project is to investigate how linear and nonlinear SVM models
perform on this regression task and how they compare to a popular tree-based
ensemble model, the Random Forest Regressor. The project aims to evaluate model
performance based on both prediction accuracy and computational efficiency, and to
provide insight into the advantages and limitations of each approach.
Four different models were trained and compared. The first model was a LinearSVR,
which is a linear version of SVM for regression tasks. This model was trained on the
entire dataset, taking only about 2 seconds to complete. However, it achieved a
relatively high RMSE of approximately 0.86, indicating underfitting. LinearSVR is known
for its speed and simplicity, but it is limited to modeling linear relationships, which
makes it unsuitable for datasets with more complex nonlinear patterns.
The second model was a standard SVR using the RBF (Radial Basis Function) kernel,
which enables the model to capture nonlinear dependencies in the data. However, due
to the nature of kernel-based SVMs, training time increases dramatically with dataset
size. To manage computational costs, this model was trained on a reduced sample of
2,500 rows. Even with the smaller dataset, the model took about 51 seconds to train
and achieved a significantly lower RMSE of around 0.58. This improved performance
demonstrates the strength of nonlinear kernels, but also highlights the model’s main
disadvantage—its poor scalability to large datasets.
To further explore the potential of SVR, a third model was built using hyperparameter
optimization with RandomizedSearchCV. This "Tuned SVR" also used only 2,500 training
samples due to computational constraints, but the training time increased to 123
seconds. Interestingly, this is longer than the time required to train a Random Forest
model on the full dataset. The best parameters found were C = 8.17, gamma = 0.0846,
and epsilon = 0.1508. The tuned model maintained an RMSE of around 0.58, similar to
the untuned SVR, suggesting that hyperparameter tuning slightly improved stability but
did not significantly enhance predictive performance.
The final model in the comparison was a Random Forest Regressor, which was trained
on the entire dataset of over 16,000 rows. Despite the large dataset, the model
completed training in approximately 100 seconds—about 20 seconds faster than the
tuned SVR, which used only a fraction of the data. This highlights Random Forest’s
efficiency and scalability. The model achieved an RMSE of around 0.49 and showed
strong performance without the need for feature scaling or kernel tricks. It was
configured with 185 estimators, no maximum depth, log2 for the maximum number of
features, and relatively low thresholds for minimum samples per split and leaf.
It is important to understand what the RMSE (Root Mean Squared Error) values mean in
this context. Since the target variable in this dataset represents median house value in
units of $100,000, an RMSE of 0.49 corresponds to an average prediction error of
approximately $49,000. This metric gives a sense of how far off the model's predictions
are from the true house prices on average. For example, if a model predicts a value of
$300,000, the actual value might typically fall in the range of about $251,000 to
$349,000. RMSE is particularly useful because it penalizes larger errors more heavily,
making it a reliable indicator of model accuracy.
Each model comes with its own strengths and weaknesses. LinearSVR is very fast and
easy to implement but limited to linear relationships, which can lead to underfitting on
complex datasets. SVR with RBF kernel provides powerful nonlinear modeling
capabilities but does not scale well with large datasets, as training time grows
exponentially with the number of samples. Tuned SVR can offer marginal performance
gains through optimized parameters but is computationally expensive and sensitive to
the tuning process. Random Forest, on the other hand, offers a strong balance of
performance, interpretability, and scalability. It is less affected by outliers and can
naturally handle nonlinearities and interactions between features. However, it can
become memory-intensive with a large number of trees and is not as fast as linear
models in deployment scenarios requiring real-time predictions.
The project was implemented as an interactive Streamlit web application. The app
allows users to explore the models and data visually, switch between models to view
their performance, and make live predictions by entering custom values for selected
features such as median income, average rooms, and house age. These input values are
combined with the average values of other features to form a complete input vector,
which is then used to predict the median house value. The app also includes summary
statistics, feature distribution plots, and a built-in chatbot that users can interact with
to ask questions about the models, the dataset, and SVM theory.
The chatbot used in this application is powered by gpt-4o-mini, a lightweight
but strong and responsive generative AI model. It is designed to provide helpful answers about the
dataset, the models, the theory behind SVM and Random Forests, and insights specific
to this application. The bot’s responses are generated using two sources: first, it
references embedded documentation that was written by the app's creator, which was
embedded using OpenAI’s text-embedding-3-small model and stored in a Chroma
vector database with around 30 indexed text chunks. Second, the chatbot can also
access live information from the app itself, such as the currently selected model, its
predictions, hyperparameters, and runtime results. This dual knowledge setup allows
the bot to answer both static questions and dynamic ones related to model outputs and
user inputs, making it an integral part of the interactive learning experience.
In conclusion, this project provides an in-depth comparison of regression techniques,
emphasizing the trade-offs between simplicity, computational cost, and predictive
power. LinearSVR offers speed but sacrifices accuracy. SVR with RBF provides high
accuracy at the cost of poor scalability. Tuned SVR allows for fine-tuned performance
but is computationally heavy and time-consuming. Random Forest proves to be the
most practical option, combining competitive accuracy with scalability and robustness,
making it a suitable model for real-world housing value prediction tasks. 
The Streamlit web application is designed with a clean and intuitive layout featuring a sidebar chatbot and three main interactive tabs.
The first tab, "Data Overview", allows users to explore the California Housing dataset visually and interactively.
Users can view the full dataset containing over 16,000 samples and eight features, including variables such as Median Income, House Age, and Average Rooms.
A checkbox reveals the raw data table, and a dropdown menu allows users to select any feature to view its distribution as a histogram. 
This is useful for gaining insights into the shape of the data and identifying skewed or multimodal distributions.
The second tab, "Model Comparison", focuses on analyzing the performance of the four regression models trained for this project: LinearSVR, SVR with RBF kernel, Tuned SVR, and Random Forest. 
Users can select any of the models from a dropdown menu, and the application displays the corresponding RMSE value, model name, and, where relevant, the best hyperparameter configuration.
 This tab is meant to help users understand how different models perform in terms of accuracy and complexity.
The third tab, "Prediction", allows users to make real-time predictions using any of the available models. 
After selecting a model, users can adjust sliders for Median Income, Average Rooms, and House Age to simulate different housing conditions. 
These inputs are combined with the average values of the remaining features, and the selected model makes a prediction based on this complete input vector. The predicted median house value is then displayed in a clear metric format. Each prediction is also stored in the app’s session state so it can be referenced by the chatbot.
The chatbot, located in the sidebar, is fully integrated with the application’s logic. 
Users can ask questions about the models, predictions, evaluation metrics, or general machine learning concepts. 
The chatbot uses two main sources of information to generate responses: embedded documentation (such as this explanation) stored in a vector database using OpenAI embeddings, and live data from the app’s session, including the most recent model used, its predictions, and the user’s input features. 
This combination enables the chatbot to provide answers that are both informative and tailored to the user’s current activity within the app.
This structure makes the application not only a powerful exploration tool for comparing regression models but also a dynamic learning platform for understanding machine learning workflows.
"""


import re

# Replace line breaks inside paragraphs with a space
clean_text = re.sub(r'\n+', ' ', full_text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", ".", " ", ""]
)

docs = splitter.create_documents([clean_text])

# === Create embedding model ===
embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# === Build Chroma DB and persist it ===
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="chroma_project_db"
)
vectorstore.persist()

print("✅ Chroma vector DB built and saved to 'chroma_project_db'.")
