import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# ==========================================
# 1. Styling & Animations
# ==========================================
st.set_page_config(page_title="Calculus AI Pro", page_icon="♾️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0052d4);
        color: white; border: none; border-radius: 10px;
        width: 100%; transition: 0.3s; font-weight: bold;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_math = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_o6spyjdz.json")

# ==========================================
# 2. Logic & API
# ==========================================
api_key = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
STABLE_MODEL = 'gemini-2.5-flash-lite'

def generate_pro_graph(equation_text, title, calc_type="General"):
    try:
        x_sym = sp.Symbol('x')
        expr = sp.sympify(equation_text)
        f_numpy = sp.lambdify(x_sym, expr, "numpy")
        x_vals = np.linspace(-10, 10, 500)
        y_vals = f_numpy(x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#00d4ff', width=4),
                                 fill='tozeroy' if calc_type == "Integration" else None,
                                 fillcolor='rgba(0, 212, 255, 0.1)', name="Curve"))
        
        # Turning Points
        derivative = sp.diff(expr, x_sym)
        critical_points = sp.solve(derivative, x_sym)
        real_crit = [float(p) for p in critical_points if p.is_real and -10 <= p <= 10]
        if real_crit:
            cp_y = [float(expr.subs(x_sym, p)) for p in real_crit]
            fig.add_trace(go.Scatter(x=real_crit, y=cp_y, mode='markers',
                                     marker=dict(color='#FF007F', size=12, symbol='diamond'), name="Critical Point"))

        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=False, zeroline=True), yaxis=dict(showgrid=False, zeroline=True),
                          margin=dict(l=0, r=0, t=30, b=0), height=350)
        return fig
    except: return None

# ==========================================
# 3. App Tabs
# ==========================================
st.title("♾️ Calculus AI Pro")
tabs = st.tabs(["⚡ Solver", "📸 Photo Math", "🎯 Exam Prep"])

# --- TAB 1: SMART SOLVER (Integrated with Similar Question Logic) ---
with tabs[0]:
    col_in, col_gr = st.columns([1, 1.5], gap="large")
    
    with col_in:
        st.markdown("### 🛠️ Controls")
        op = st.selectbox("Select Action", ["Differentiation", "Integration"], key="solver_op")
        val = st.text_input("Enter Function", "x**2 - 4*x", key="solver_in")
        
        btn_solve = st.button("Solve & Explain ✨", type="primary")
        btn_similar = st.button("Practice Similar Problem 🔄")

    with col_gr:
        fig = generate_pro_graph(val, f"f(x) = {val}", op)
        if fig: 
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a valid mathematical function to see the visualization.")

    # 1. Main Solution Logic
    if btn_solve:
        st.markdown("---")
        try:
            # Mathematical verification using SymPy
            x_sym = sp.Symbol('x')
            expr = sp.sympify(val)
            res = sp.diff(expr, x_sym) if op == "Differentiation" else sp.integrate(expr, x_sym)
            
            st.latex(f"Result: {sp.latex(res)}")
            
            with st.spinner("AI is drafting the steps..."):
                try:
                    prompt = f"Explain step-by-step how to {op} the function {val} for a Class 12 student. The verified answer is {res}."
                    resp = client.models.generate_content(model=STABLE_MODEL, contents=prompt)
                    st.write(resp.text)
                except Exception as ai_limit_err:
                    st.warning("⚠️ AI is cooling down. The math above is correct, but please wait 10 seconds for the step-by-step explanation.")
        except Exception as math_err:
            st.error(f"❌ Math Error: {math_err}. Check your syntax (e.g., use 5*x instead of 5x).")

    # 2. Similar Question Logic (State-Managed)
    if btn_similar:
        with st.spinner("Creating a twin challenge..."):
            try:
                sim_prompt = f"Generate one similar Class 12 math problem to {val} for {op}. Provide only the question text."
                sim_q = client.models.generate_content(model=STABLE_MODEL, contents=sim_prompt)
                st.session_state['sim_question'] = sim_q.text
                # Reset the solution state when a new question is born
                if 'sim_solution' in st.session_state: 
                    del st.session_state['sim_solution']
            except Exception as e:
                st.error("⚠️ Quota reached. Please wait a moment before generating another practice problem.")

    # 3. Persistent Display for Practice
    if 'sim_question' in st.session_state:
        st.markdown("---")
        st.markdown("### 📝 Practice Zone")
        st.info(f"**Try this similar problem:** {st.session_state['sim_question']}")
        
        # Reveal Solution Button
        if st.button("Check Answer for Similar Problem"):
            with st.spinner("Solving practice problem..."):
                try:
                    sim_s_prompt = f"Provide a detailed step-by-step LaTeX solution for: {st.session_state['sim_question']}"
                    sim_s = client.models.generate_content(model=STABLE_MODEL, contents=sim_s_prompt)
                    st.session_state['sim_solution'] = sim_s.text
                except Exception:
                    st.error("⚠️ AI limit exceeded. Please wait 15 seconds and try revealing the answer again.")
        
        # Display the solution if it exists in memory
        if 'sim_solution' in st.session_state:
            st.success("### Practice Solution")
            st.write(st.session_state['sim_solution'])
# --- OPTIMIZED TAB 2: PHOTO MATH ---
if st.button("Visual Analysis & Solve"):
    with st.spinner("Analyzing handwriting..."):
        try:
            # We ask for BOTH things in one single request to save quota
            master_prompt = """
            1. Solve this problem step-by-step using LaTeX.
            2. At the very end of your response, provide the raw Python-compatible 
               formula for this function on a single line starting with 'FORMULA:' 
               (e.g., FORMULA: x**3 + sin(x)).
            """
            
            response = client.models.generate_content(model=STABLE_MODEL, contents=[img, master_prompt])
            full_text = response.text
            
            # Split the response to get the solution and the formula
            if "FORMULA:" in full_text:
                solution_part, formula_part = full_text.split("FORMULA:")
                f_text = formula_part.strip().replace('`','')
            else:
                solution_part = full_text
                f_text = "x" # Fallback

            col_s, col_g = st.columns([1, 1])
            with col_s:
                st.markdown("### Solution")
                st.write(solution_part)
            
            with col_g:
                fig = generate_elaborate_graph(f_text, f"Visualizing: {f_text}")
                if fig: st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"⚠️ API Error: {str(e)}")
# ==========================================
# 1. Clean Sidebar (Universal)
# ==========================================
with st.sidebar:
    if lottie_math: 
        st_lottie(lottie_math, height=120, key="sidebar_anim")
    st.title("♾️ Calculus Lab")
    st.markdown("---")
    st.markdown("### Fundamental Rules")
    st.latex(r"\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)") # Chain Rule
    st.latex(r"\int u \, dv = uv - \int v \, du")    # Integration by Parts
    st.markdown("---")
    st.info("Advanced Mathematical Engine for Differentiation, Integration, and Limits.")

# ==========================================
# 2. Updated Tabs (Universal Logic)
# ==========================================
# In Tab 1, 2, and 3, I've updated the prompts to be "Level-Agnostic"

# --- TAB 3: MASTER PRACTICE ---
with tabs[2]:
    st.markdown("### 🎯 Challenge Generator")
    # Higher-level topics added
    exam_topic = st.selectbox("Select Domain", 
                              ["Differential Calculus", "Integral Calculus", "Multivariable Concepts", "Differential Equations"])
    
    if st.button("Generate Advanced Challenge 🎯"):
        # The prompt now asks for professional-level questions
        q = client.models.generate_content(
            model=STABLE_MODEL, 
            contents=f"Generate a challenging university-level calculus question about {exam_topic}. Provide only the question."
        )
        st.session_state['exam_q'] = q.text
        
    if 'exam_q' in st.session_state:
        st.info(st.session_state['exam_q'])
        if st.button("Reveal Mathematical Proof"):
            s = client.models.generate_content(
                model=STABLE_MODEL, 
                contents=f"Provide a rigorous, step-by-step mathematical solution for: {st.session_state['exam_q']}. Use LaTeX."
            )
            st.success(s.text)
