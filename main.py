import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# ==========================================
# 1. Page & Aesthetic Config
# ==========================================
st.set_page_config(page_title="Calculus AI Pro", page_icon="♾️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0052d4);
        color: white; border: none; border-radius: 10px; width: 100%; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. API & Universal Graphing
# ==========================================
api_key = st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
# Using Flash-Lite for speed and lower token cost
MODEL_ID = 'gemini-2.0-flash-lite' 

def generate_graph(equation_text, calc_type="General"):
    try:
        x_sym = sp.Symbol('x')
        expr = sp.sympify(equation_text)
        f_numpy = sp.lambdify(x_sym, expr, "numpy")
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f_numpy(x_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#00d4ff', width=4),
                                 fill='tozeroy' if calc_type == "Integration" else None,
                                 fillcolor='rgba(0, 212, 255, 0.1)'))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, t=10, b=0), height=350)
        return fig
    except: return None

# ==========================================
# 3. Universal App Tabs
# ==========================================
st.title("♾️ Calculus AI Pro")
tabs = st.tabs(["⚡ Solver", "📸 Photo Math", "🎯 Practice"])

# --- TAB 1: SOLVER ---
with tabs[0]:
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        op = st.selectbox("Action", ["Differentiation", "Integration"], key="s_op")
        val = st.text_input("Equation", "x**2 - 4*x", key="s_in")
        btn_solve = st.button("Solve ✨")
        btn_sim = st.button("Similar Problem 🔄")

    with c2:
        fig = generate_graph(val, op)
        if fig: st.plotly_chart(fig, use_container_width=True)

    if btn_solve:
        st.markdown("---")
        try:
            # Symbolic Math (Uses 0 API Tokens)
            res = sp.diff(sp.sympify(val), sp.Symbol('x')) if op == "Differentiation" else sp.integrate(sp.sympify(val), sp.Symbol('x'))
            st.latex(f"Result: {sp.latex(res)}")
            
            # AI Step-by-Step (Short & Sweet)
            prompt = f"Solve {op} for {val}. BRIEF LaTeX steps only. No intro/outro."
            resp = client.models.generate_content(model=MODEL_ID, contents=prompt, config={'temperature': 0.1})
            st.write(resp.text)
        except Exception: st.warning("Math solved, but AI explanation is cooling down. Wait 15s.")

    if btn_sim:
        st.markdown("---")
        try:
            sim = client.models.generate_content(model=MODEL_ID, contents=f"Provide one similar math problem to {val} for {op}. Question only.")
            st.session_state['sq'] = sim.text
            if 'ss' in st.session_state: del st.session_state['ss']
        except: st.error("Quota full. Wait a moment.")

    if 'sq' in st.session_state:
        st.info(f"**Try this:** {st.session_state['sq']}")
        if st.button("Check Practice Answer"):
            ans = client.models.generate_content(model=MODEL_ID, contents=f"Brief LaTeX solution for: {st.session_state['sq']}")
            st.session_state['ss'] = ans.text
        if 'ss' in st.session_state: st.success(st.session_state['ss'])

# --- TAB 2: PHOTO MATH (Single Call Logic) ---
with tabs[1]:
    up = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if up:
        img = Image.open(up)
        if st.button("Analyze & Solve 📸"):
            try:
                # Optimized Single Call
                p = "Solve step-by-step using LaTeX. End with 'FORMULA: [raw python equation]'."
                r = client.models.generate_content(model=MODEL_ID, contents=[img, p], config={'temperature': 0.1})
                txt = r.text
                
                sol_part = txt.split("FORMULA:")[0] if "FORMULA:" in txt else txt
                f_part = txt.split("FORMULA:")[1].strip().replace('`','') if "FORMULA:" in txt else "x"
                
                sc1, sc2 = st.columns([1, 1])
                with sc1: st.write(sol_part)
                with sc2:
                    fig = generate_graph(f_part)
                    if fig: st.plotly_chart(fig, use_container_width=True)
            except: st.error("AI is busy. Please retry in 20 seconds.")

# --- TAB 3: PRACTICE (Level Agnostic) ---
with tabs[2]:
    topic = st.selectbox("Topic", ["Differentiation", "Integration", "Limits", "Derivatives Application"])
    if st.button("Get Advanced Challenge"):
        try:
            q = client.models.generate_content(model=MODEL_ID, contents=f"One hard {topic} question. Question only.")
            st.session_state['pq'] = q.text
        except: st.error("Wait a moment for new questions.")
    
    if 'pq' in st.session_state:
        st.info(st.session_state['pq'])
        if st.button("Reveal Proof"):
            try:
                s = client.models.generate_content(model=MODEL_ID, contents=f"Brief LaTeX solution for: {st.session_state['pq']}")
                st.write(s.text)
            except: st.warning("Limit reached. Please wait.")
