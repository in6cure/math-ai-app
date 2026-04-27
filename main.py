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

# --- TAB 1: SOLVER ---
with tabs[0]:
    col_in, col_gr = st.columns([1, 1.5], gap="large")
    
    with col_in:
        st.markdown("### 🛠️ Controls")
        op = st.selectbox("Select Action", ["Differentiation", "Integration"], key="solver_op")
        val = st.text_input("Enter Function", "x**2 - 4*x", key="solver_in")
        
        btn_solve = st.button("Solve & Explain ✨")
        btn_similar = st.button("Practice Similar Problem 🔄")

    with col_gr:
        fig = generate_pro_graph(val, f"f(x) = {val}", op)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # 1. Main Solution Logic
    if btn_solve:
        st.markdown("---")
        try:
            res = sp.diff(sp.sympify(val), sp.Symbol('x')) if op == "Differentiation" else sp.integrate(sp.sympify(val), sp.Symbol('x'))
            st.latex(f"Result: {sp.latex(res)}")
            resp = client.models.generate_content(model=STABLE_MODEL, contents=f"Explain step-by-step how to {op} {val}. Result is {res}.")
            st.write(resp.text)
        except Exception as e: st.error(f"Error: {e}")

    # 2. Similar Question Logic (Interactive)
    if btn_similar:
        with st.spinner("Generating a similar challenge..."):
            sim_q = client.models.generate_content(model=STABLE_MODEL, contents=f"Generate one similar Class 12 problem to {val} for {op}. Provide only the question.")
            st.session_state['sim_question'] = sim_q.text
            # Clear old solution when new question is generated
            if 'sim_solution' in st.session_state: del st.session_state['sim_solution']

    # Display the Similar Question and its Solution Reveal button
    if 'sim_question' in st.session_state:
        st.markdown("---")
        st.info(f"**Try this similar problem:** {st.session_state['sim_question']}")
        
        if st.button("Check Answer for Similar Problem"):
            with st.spinner("Solving practice problem..."):
                sim_s = client.models.generate_content(model=STABLE_MODEL, contents=f"Solve this math problem step-by-step: {st.session_state['sim_question']}")
                st.session_state['sim_solution'] = sim_s.text
        
        if 'sim_solution' in st.session_state:
            st.success("### Practice Solution")
            st.write(st.session_state['sim_solution'])
# --- TAB 2: PHOTO ---
with tabs[1]:
    up = st.file_uploader("Upload Problem", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        c_img, c_sol = st.columns([1, 2])
        with c_img: st.image(img, use_container_width=True)
        with c_sol:
            if st.button("AI Vision Solve 📸"):
                sol = client.models.generate_content(model=STABLE_MODEL, contents=[img, "Solve this math problem step-by-step using LaTeX."])
                st.write(sol.text)

# --- TAB 3: PRACTICE ---
with tabs[2]:
    st.markdown("### 🏆 Exam Prep Center")
    exam_topic = st.selectbox("What do you want to practice?", 
                              ["Differentiation Basics", "Application of Derivatives", "Definite Integrals", "Area Under Curve"])
    
    if st.button("Generate Board-Level Question 🎯"):
        q = client.models.generate_content(model=STABLE_MODEL, contents=f"Generate a difficult Class 12 Board exam question about {exam_topic}.")
        st.session_state['exam_q'] = q.text
        
    if 'exam_q' in st.session_state:
        st.info(st.session_state['exam_q'])
        if st.button("Show Step-by-Step Solution"):
            s = client.models.generate_content(model=STABLE_MODEL, contents=f"Solve: {st.session_state['exam_q']} step-by-step for Class 12.")
            st.success(s.text)

# Sidebar
with st.sidebar:
    if lottie_math: st_lottie(lottie_math, height=120)
    st.title("📚 Study Lab")
    st.markdown("---")
    st.latex(r"\frac{d}{dx}x^n = nx^{n-1}")
    st.info("Bikaner's Best AI Math Tutor")
