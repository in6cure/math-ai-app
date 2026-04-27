import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# ==========================================
# 1. Aesthetic Styling & Glassmorphism
# ==========================================
st.set_page_config(page_title="Calculus AI Pro", page_icon="♾️", layout="wide")

# This creates the neon glow and gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0052d4);
        color: white;
        border: none;
        border-radius: 10px;
        transition: 0.3s;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie Animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_math = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_o6spyjdz.json")

# ==========================================
# 2. Sidebar & API Setup
# ==========================================
with st.sidebar:
    if lottie_math:
        st_lottie(lottie_math, height=150, key="math_sidebar")
    st.title("📚 Study Lab")
    st.markdown("---")
    st.latex(r"\frac{d}{dx}x^n = nx^{n-1}")
    st.latex(r"\int x^n dx = \frac{x^{n+1}}{n+1}")
    st.info("AI is ready to assist with your Class 12 Boards!")

api_key = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
STABLE_MODEL = 'gemini-2.5-flash-lite'

# ==========================================
# 3. Interactive Graphing Engine
# ==========================================
def generate_pro_graph(equation_text, title, calc_type="General"):
    try:
        x_sym = sp.Symbol('x')
        expr = sp.sympify(equation_text)
        f_numpy = sp.lambdify(x_sym, expr, "numpy")
        x_vals = np.linspace(-10, 10, 500)
        y_vals = f_numpy(x_vals)

        fig = go.Figure()
        # Main Line with Neon Glow
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines',
            line=dict(color='#00d4ff', width=5),
            fill='tozeroy' if calc_type == "Integration" else None,
            fillcolor='rgba(0, 212, 255, 0.1)',
            name="Curve"
        ))
        
        # Add animated markers for turning points
        derivative = sp.diff(expr, x_sym)
        critical_points = sp.solve(derivative, x_sym)
        real_crit = [float(p) for p in critical_points if p.is_real and -10 <= p <= 10]
        if real_crit:
            cp_y = [float(expr.subs(x_sym, p)) for p in real_crit]
            fig.add_trace(go.Scatter(
                x=real_crit, y=cp_y, mode='markers',
                marker=dict(color='#FF007F', size=15, symbol='diamond', line=dict(width=2, color="white")),
                name="Critical Points"
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white'),
            yaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white'),
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
        return fig
    except: return None

# ==========================================
# 4. App Interface
# ==========================================
st.title("♾️ Calculus AI Pro")
st.caption("Advanced Math Visualization Engine v2.0")

tabs = st.tabs(["⚡ Solver", "📸 Photo Math", "🎯 Exam Prep"])

# --- TAB 1: SOLVER ---
with tabs[0]:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("### Input")
        op = st.selectbox("Operation", ["Differentiation", "Integration"], key="op1")
        val = st.text_input("Equation", "x**2 - 4*x", key="in1")
        if st.button("Generate Magic ✨", key="btn1"):
            try:
                res = sp.diff(sp.sympify(val), sp.Symbol('x')) if op == "Differentiation" else sp.integrate(sp.sympify(val), sp.Symbol('x'))
                st.latex(f"Result: {sp.latex(res)}")
                resp = client.models.generate_content(model=STABLE_MODEL, contents=f"Solve {val} step-by-step for Class 12. Result is {res}. Use LaTeX.")
                st.write(resp.text)
            except Exception as e: st.error(e)
    with c2:
        fig = generate_pro_graph(val, f"f(x) = {val}", op)
        if fig: st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PHOTO ---
with tabs[1]:
    up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        st.image(img, width=300)
        if st.button("Solve from Image 📸"):
            with st.spinner("Decoding handwriting..."):
                sol = client.models.generate_content(model=STABLE_MODEL, contents=[img, "Solve this math problem step-by-step using LaTeX."])
                form = client.models.generate_content(model=STABLE_MODEL, contents=[img, "Return only the Python equation like x**2."])
                sc1, sc2 = st.columns(2)
                with sc1: st.write(sol.text)
                with sc2:
                    fig = generate_pro_graph(form.text.strip(), "Handwritten Analysis")
                    if fig: st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: PRACTICE ---
with tabs[2]:
    if st.button("Get Daily Challenge 🎯"):
        q = client.models.generate_content(model=STABLE_MODEL, contents="Give one hard Class 12 calculus board exam question.")
        st.session_state['active_q'] = q.text
    if 'active_q' in st.session_state:
        st.info(st.session_state['active_q'])
        if st.button("Show Solution"):
            s = client.models.generate_content(model=STABLE_MODEL, contents=f"Solve: {st.session_state['active_q']} step-by-step.")
            st.success(s.text)
