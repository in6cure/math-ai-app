import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image

# 1. Page Config & Professional Sidebar
st.set_page_config(page_title="Calculus Master Pro AI", page_icon="♾️", layout="wide")

with st.sidebar:
    st.title("📚 Study Materials")
    st.markdown("### Formula Cheat Sheet")
    st.latex(r"\frac{d}{dx}x^n = nx^{n-1}")
    st.latex(r"\int x^n dx = \frac{x^{n+1}}{n+1} + C")
    st.latex(r"\frac{d}{dx}\sin(x) = \cos(x)")
    st.markdown("---")
    st.info("Tip: Use ** for powers (x**2) and * for multiply (5*x).")

st.title("♾️ Calculus Master Pro AI")
st.subheader("Your Class 12 & CUET Math Partner")

# 2. API Setup
api_key = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 3. Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["🧮 Solver & Grapher", "📸 Photo Math", "📝 Exam Practice"])

# --- TAB 1: SOLVER & GRAPHER ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        calc_type = st.selectbox("Operation", ["Differentiation", "Integration"])
        user_input = st.text_input("Enter Function:", value="x**2")
        
        if st.button("Solve & Graph"):
            x_sym = sp.Symbol('x')
            expr = sp.sympify(user_input)
            
            # Logic
            if calc_type == "Differentiation":
                res_sym = sp.diff(expr, x_sym)
                label = "Derivative"
            else:
                res_sym = sp.integrate(expr, x_sym)
                label = "Integral"
            
            st.latex(f"Result: {sp.latex(res_sym)}")
            
            # AI Explanation
            prompt = f"Explain the {calc_type} of {user_input} step-by-step for a student. Result is {res_sym}."
            resp = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
            st.markdown(resp.text)

    with col2:
        # Simple Plotly Graph
        try:
            f = sp.lambdify(sp.Symbol('x'), sp.sympify(user_input), "numpy")
            x_vals = np.linspace(-10, 10, 100)
            y_vals = f(x_vals)
            fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, name="Function"))
            fig.update_layout(title="Function Preview", height=400)
            st.plotly_chart(fig)
        except:
            st.write("Graph not available for this input.")

# --- TAB 2: PHOTO MATH (OCR) ---
with tab2:
    st.write("Upload a photo of your handwritten math problem.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Problem", width=300)
        
        if st.button("Read & Solve Image"):
            with st.spinner("AI is reading your handwriting..."):
                # Sending image to Gemini 1.5 Flash (best for OCR)
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[img, "Identify the math problem in this image and solve it step-by-step with LaTeX."]
                )
                st.markdown(response.text)

# --- TAB 3: EXAM PRACTICE ---
with tab3:
    st.write("Generate a random Class 12 / CUET style question.")
    topic = st.selectbox("Topic", ["Calculus Basics", "Applications of Derivatives", "Definite Integrals"])
    
    if st.button("Generate Random Question"):
        prompt = f"Generate one difficult Class 12 exam question about {topic}. Provide only the question first."
        resp = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        st.info(resp.text)
        
        if st.button("Show Solution"):
            sol_prompt = f"Provide a detailed solution for: {resp.text}"
            sol_resp = client.models.generate_content(model='gemini-1.5-flash', contents=sol_prompt)
            st.success(sol_resp.text)
