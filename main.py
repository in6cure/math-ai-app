import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image

# 1. Page Configuration & Professional Sidebar
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

# 2. API Setup with Error Handling
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key missing! Go to Settings -> Secrets and add GEMINI_API_KEY")
    st.stop()

# Initializing the client
client = genai.Client(api_key=api_key)

# 3. Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["🧮 Solver & Grapher", "📸 Photo Math", "📝 Exam Practice"])

# --- TAB 1: SOLVER & GRAPHER ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        calc_type = st.selectbox("Operation", ["Differentiation", "Integration"])
        user_input = st.text_input("Enter Function (e.g., x**2 + 5*x):", value="x**2")
        
        if st.button("Solve & Graph"):
            try:
                x_sym = sp.Symbol('x')
                expr = sp.sympify(user_input)
                
                # Math Logic
                if calc_type == "Differentiation":
                    res_sym = sp.diff(expr, x_sym)
                else:
                    res_sym = sp.integrate(expr, x_sym)
                
                st.success("Mathematical Calculation Successful!")
                st.latex(f"Result: {sp.latex(res_sym)}")
                
                # --- INTEGRATED DEBUG AI LOGIC ---
                st.markdown("### Step-by-Step Logic")
                with st.spinner("AI is explaining the steps..."):
                    try:
                        prompt = f"Explain the {calc_type} of {user_input} step-by-step for a Class 12 student. The verified answer is {res_sym}. Use LaTeX."
                        resp = client.models.generate_content(
                            model='gemini-2.5-flash-lite', 
                            contents=prompt
                        )
                        st.markdown(resp.text)
                    except Exception as ai_err:
                        st.error(f"⚠️ GOOGLE API ERROR: {str(ai_err)}")
                        st.info("Check if your API key is active or if you have reached your free limit.")

            except Exception as math_err:
                st.error(f"❌ Math Syntax Error: {math_err}. Make sure to use * for multiplication.")

    with col2:
        try:
            # Create a graph of the original function
            f_graph = sp.lambdify(sp.Symbol('x'), sp.sympify(user_input), "numpy")
            x_vals = np.linspace(-10, 10, 100)
            y_vals = f_graph(x_vals)
            
            fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals, mode='lines', name="f(x)", line=dict(color='royalblue', width=4)))
            fig.update_layout(title=f"Graph of {user_input}", xaxis_title="x", yaxis_title="f(x)", height=450)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Could not generate graph. Try a simpler function like x**2.")

# --- TAB 2: PHOTO MATH (OCR) ---
with tab2:
    st.write("Upload a photo of your handwritten math problem.")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Problem to Solve", width=400)
        
        if st.button("Read & Solve with Gemini"):
            with st.spinner("Analyzing image..."):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=[img, "Analyze this image. Identify the math problem and provide a full step-by-step solution in LaTeX."]
                    )
                    st.markdown("### Solution from Photo")
                    st.markdown(response.text)
                except Exception as img_err:
                    st.error(f"⚠️ IMAGE AI ERROR: {str(img_err)}")

# --- TAB 3: EXAM PRACTICE ---
with tab3:
    st.write("Generate custom practice problems for CUET or Board Exams.")
    topic = st.selectbox("Select Topic", ["Calculus Basics", "Maxima and Minima", "Definite Integrals", "Rate of Change"])
    
    if st.button("Generate Practice Question"):
        try:
            q_prompt = f"Generate a high-level Class 12 board exam question about {topic}. Do not provide the solution yet."
            q_resp = client.models.generate_content(model='gemini-2.5-flash-lite', contents=q_prompt)
            st.info(f"**Your Question:**\n\n{q_resp.text}")
            
            # Store the question in a hidden way to solve it if requested
            st.session_state['current_q'] = q_resp.text
        except Exception as e:
            st.error(f"Error generating question: {str(e)}")

    if 'current_q' in st.session_state:
        if st.button("Reveal Detailed Solution"):
            try:
                s_prompt = f"Provide a step-by-step LaTeX solution for this question: {st.session_state['current_q']}"
                s_resp = client.models.generate_content(model='gemini-2.5-flash-lite', contents=s_prompt)
                st.success("### Step-by-Step Answer")
                st.markdown(s_resp.text)
            except Exception as e:
                st.error(f"Error generating solution: {str(e)}")
