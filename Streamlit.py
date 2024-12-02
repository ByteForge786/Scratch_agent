import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from typing import Dict, Any
import json

def load_schema_config():
    with open("schema_config.yaml") as f:
        return yaml.safe_load(f)

def create_streamlit_app():
    st.set_page_config(page_title="Text to SQL Analytics", layout="wide")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Load configuration
    config = load_schema_config()
    
    # Main interface
    st.title("Natural Language SQL Analytics")
    st.write("Ask questions about your business data in natural language!")
    
    # Query input
    query = st.text_input("Enter your question:")
    
    if st.button("Analyze"):
        # Process query
        agent = TextToSQLAgent("google/flan-t5-base", config)
        result = agent.process_query(query)
        
        if result["status"] == "success":
            # Store in history
            st.session_state.history.append({
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
    
    # Display history with expandable sections
    st.header("Analysis History")
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Query: {item['query']} ({item['timestamp']})"):
            # Show execution flow
            st.subheader("Execution Flow")
            for step in item['result']['execution_flow']:
                st.write(f"Step: {step['step']}")
                st.write(f"Explanation: {step['details'].get('explanation', '')}")
            
            # Show SQL
            st.subheader("Generated SQL")
            st.code(item['result']['sql'], language="sql")
            
            # Show results
            st.subheader("Results")
            df = pd.DataFrame(item['result']['results']['data'], 
                            columns=item['result']['results']['columns'])
            st.dataframe(df)
            
            # Show visualization if available
            if 'visualization' in item['result']:
                st.subheader("Visualization")
                st.plotly_chart(item['result']['visualization']['figure'])
                
                # Show visualization explanation
                with st.expander("Why this visualization?"):
                    st.write(item['result']['visualization']['explanation'])

if __name__ == "__main__":
    create_streamlit_app()
