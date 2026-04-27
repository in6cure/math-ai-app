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

# --- TAB 2: PHOTO MATH (OCR & SOLVE) ---
with tab2:
    st.header("📸 Photo Math Solver")
    st.write("Upload a photo of your handwritten math problem or a textbook page.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Choose a JPG, PNG, or JPEG image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Problem to Solve", width=400)
        
        # 2. Solve Button
        if st.button("Read & Solve with Gemini"):
            with st.spinner("AI is analyzing the image and solving..."):
                try:
                    # THE UPDATED PROMPT:
                    # We tell the AI to be a tutor and avoid "Code" formatting.
                    image_prompt = """
                    Identify the mathematical problem in this image and solve it step-by-step.
                    
                    RULES:
                    1. Use clear Class 12 mathematical logic.
                    2. Use standard Markdown and LaTeX (e.g., $...$ or $$...$$) for all math.
                    3. DO NOT include LaTeX document headers (no \documentclass, no \begin{document}).
                    4. If the image is blurry or the math is unclear, ask the user to retake the photo.
                    5. Start directly with 'Problem Identified:'
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=[img, image_prompt]
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown("### Solution from Photo")
                    # Using st.write to ensure LaTeX renders properly
                    st.write(response.text)
                    
                except Exception as img_err:
                    # DEBUG LOGIC:
                    st.error(f"⚠️ IMAGE AI ERROR: {str(img_err)}")
                    st.info("Ensure your API key is correct and you haven't exceeded your free limit.")
    else:
        st.info("Waiting for image upload... Once uploaded, click the button above to solve.")


# --- TAB 3: EXAM PRACTICE ---
with tab3:
    st.write("Generate custom practice problems for CUET or Board Exams.")
    topic = st.selectbox("Select Topic", ["Calculus Basics", "Maxima and Minima", "Definite Integrals", "Rate of Change"])
    
    if st.button("Generate Random Question"):
        try:
            # We add a hint here too so the question itself is clean
            q_prompt = f"Generate a Class 12 board exam question about {topic}. Provide only the question text in clean Markdown."
            q_resp = client.models.generate_content(model='gemini-2.5-flash-lite', contents=q_prompt)
            st.session_state['current_q'] = q_resp.text
            if 'current_sol' in st.session_state:
                del st.session_state['current_sol']
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if 'current_q' in st.session_state:
        st.info(f"**Question:**\n\n{st.session_state['current_q']}")
        
        if st.button("Reveal Detailed Solution"):
            try:
                with st.spinner("Calculating..."):
                    # THIS IS YOUR UPDATED PROMPT
                    s_prompt = f"""
                    Provide a step-by-step math solution for this question: {st.session_state['current_q']}
                    IMPORTANT: 
                    1. Use standard Markdown and LaTeX (e.g., $...$ or $$...$$).
                    2. DO NOT include LaTeX document headers like \documentclass or \begin{{document}}. 
                    3. Start directly with 'Step 1:'
                    """
                    s_resp = client.models.generate_content(model='gemini-2.5-flash-lite', contents=s_prompt)
                    st.session_state['current_sol'] = s_resp.text
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if 'current_sol' in st.session_state:
        st.success("### Step-by-Step Answer")
        # st.write handles LaTeX better than raw st.markdown in some cases
        st.write(st.session_state['current_sol'])
