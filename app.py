import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import joblib
import plotly.express as px
import pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# --- Load Trained Model and Preprocessor ---
try:
    model = joblib.load('price_prediction_model.joblib')
    preprocessor = joblib.load('price_preprocessor.joblib')
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please run the training script first.")
    st.stop()

# --- Pinecone Setup ---
PINECONE_API_KEY = "USEYOURS"
PINECONE_ENVIRONMENT = "USEYORUS"
PINECONE_INDEX_NAME = "USEYOURS"
NAMESPACE = "USEYOURS"

# --- Embedding Model ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# --- Initialize Pinecone Connection ---
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, if not, create it
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)

# --- Generate Sample Data ---
def generate_single_data(product_type):
    payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking']
    timestamp = datetime.now()
    price = round(np.random.uniform(10, 500), 2)
    num_clicks = np.random.randint(1, 100)
    payment_method = np.random.choice(payment_methods)
    customer_id = np.random.randint(1000, 9999)
    return pd.DataFrame([{
        'timestamp': timestamp,
        'product_type': product_type,
        'price': price,
        'num_clicks': num_clicks,
        'payment_method': payment_method,
        'customer_id': customer_id
    }])

def predict_price(data):
    predict_df = data[['product_type', 'num_clicks']]
    processed_data = preprocessor.transform(predict_df)
    processed_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out(['product_type', 'num_clicks']))
    prediction = model.predict(processed_df)
    return prediction[0]

# --- Pinecone Search ---
def search_pinecone(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0]
    try:
        results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True, namespace=NAMESPACE)
        return results
    except Exception as e:
        st.error(f"Error during Pinecone search: {e}")
        return {}

# --- QA Response Generator ---
def generate_response(query, context, llm_model=""):
    try:
        pipe = pipeline("question-answering", model=llm_model)
        result = pipe(question=query, context=context)
        return result['answer'].strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I could not generate a response."

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Real-Time Sales & Chatbot", layout="wide")
    st.title(" Real-Time Sales Dashboard +  Chatbot")

    tabs = st.tabs(["📈 Dashboard", "💬 Chatbot"])

    with tabs[0]:
        st.sidebar.header("⚙️ Controls")
        simulation_speed = st.sidebar.slider("Simulation Speed (seconds)", 0.1, 5.0, 1.0)
        category_filter = st.sidebar.selectbox("Choose a Product Type", ['Electronics', 'Clothing', 'Books', 'Home Goods'])

        st.header("Real-Time Price Prediction")

        
        if 'data_history' not in st.session_state:
            st.session_state.data_history = pd.DataFrame(columns=['timestamp', 'product_type', 'price', 'num_clicks', 'predicted_price'])

        auto_simulation = st.sidebar.checkbox("Auto Simulate Data", value=True)

        if auto_simulation:
            # Simulation running logic without rerun
            time.sleep(simulation_speed)  # Delay based on user speed control

            # Generate and predict new data
            new_data = generate_single_data(category_filter)
            predicted_price = predict_price(new_data)
            new_data['predicted_price'] = predicted_price

            # Add to session state data
            st.session_state.data_history = pd.concat([st.session_state.data_history, new_data], ignore_index=True)
            st.session_state.data_history['timestamp'] = pd.to_datetime(st.session_state.data_history['timestamp'])

            # Sort and keep only the most recent 20 data points
            st.session_state.data_history = st.session_state.data_history.sort_values(by='timestamp', ascending=False).head(20)

        st.subheader("📈 Live Simulated Data and Predictions")
        if not st.session_state.data_history.empty:
            st.dataframe(st.session_state.data_history[['timestamp', 'product_type', 'price', 'num_clicks', 'predicted_price']])
        
            # Plotting the predicted price
            fig = px.line(
                st.session_state.data_history.sort_values(by='timestamp'),
                x='timestamp',
                y='predicted_price',
                title='Predicted Price Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available to display.")

    with tabs[1]:
        st.header("🧾 Document Q&A Chatbot")
        query = st.text_input("Ask a question about the documentation:")
        if query:
            with st.spinner("🔍 Searching documents..."):
                search_results = search_pinecone(query)
                context = ""
                for match in search_results.get('matches', []):
                    context += f"{match['metadata']['source']} (Chunk {match['metadata']['chunk']}): {match['metadata']['text']}\n---\n"

            st.subheader("📚 Retrieved Context")
            st.info(context if context else "No relevant documents found.")

            if context:
                llm_model_name = "distilbert-base-cased-distilled-squad"
                with st.spinner("💡 Generating response..."):
                    response = generate_response(query, context, llm_model_name)
                    st.subheader("💬 Chatbot Response:")
                    st.success(response)

if __name__ == "__main__":
    main()
