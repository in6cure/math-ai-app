import os
import streamlit as st
import sympy as sp
from google import genai

# 1. Page Configuration
st.set_page_config(page_title="Exam Logic Engine", page_icon="🧠", layout="centered")
st.title("📈 Economics & Maths Logic Engine")
st.write("Enter a Total Cost function to find Marginal Cost step-by-step.")

# 2. Setup API Client
# On Streamlit Cloud, this pulls from your 'Secrets' menu
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("API Key missing! Please add GEMINI_API_KEY to your Streamlit Cloud Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 3. User Input
user_input = st.text_input("Enter Total Cost Function (e.g., x**3 - 4*x**2 + 6*x):", value="x**3 - 4*x**2 + 6*x")

# 4. The "Solve" Button
if st.button("Solve Step-by-Step"):
    with st.spinner("Calculating exact math and generating logic steps..."):
        try:
            # --- MATH ENGINE ---
            x = sp.Symbol('x')
            # SymPy does the actual derivative here
            marginal_cost = sp.diff(sp.sympify(user_input), x)
            
            # --- AI EXPLAINER ---
            prompt = f"""
            You are an expert Economics and Maths tutor. 
            A student has a Total Cost function of: {user_input}
            They need to find the Marginal Cost. 
            The mathematically verified final answer is: {marginal_cost}

            Provide a structured, step-by-step mathematical solution to reach this answer. 
            Use LaTeX for all math formatting. Keep it clear and logical for a Class 12 student.
            """
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            
            # 5. Display the Results
            st.success("Solution Generated!")
            st.markdown("### Step-by-Step Breakdown")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error: {e}. Please check your equation format.")
          
