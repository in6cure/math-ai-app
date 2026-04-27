import streamlit as st
import sympy as sp
from google import genai
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# ==========================================
# 1. Page Config & Professional Styling
# ==========================================
st.set_page_config(page_title="Calculus Master Pro AI", page_icon="♾️", layout="wide")

# Custom CSS for a more "Interesting" Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    </style>
    """, unsafe_allow_stdio=True)

with st.sidebar:
    st.title("📚 Study Materials")
    st.markdown("### Formula Cheat Sheet")
    st.latex(r"\frac{d}{dx}x^n = nx^{n-1}")
    st.latex(r"\int x^n dx = \frac{x^{n+1}}{n+1} + C")
    st.latex(r"\frac{d}{dx}\sin(x) = \cos(x)")
    st.markdown("---")
    st.info("Tip: Use ** for powers (x**2) and * for multiply (5*x).")

st.title("♾️ Calculus Master Pro AI")
st.subheader("Your AI-Powered Math & Economics Lab")

# ==========================================
# 2. API Setup & ELABORATIVE GRAPHING ENGINE
# ==========================================
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("❌ API Key missing!")
    st.stop()
client = genai.Client(api_key=api_key)
STABLE_MODEL = 'gemini-2.5-flash-lite' 

def generate_elaborate_graph(equation_text, title_text, calc_type="General"):
    try:
        x_sym = sp.Symbol('x')
        expr = sp.sympify(equation_text)
        f_numpy = sp.lambdify(x_sym, expr, "numpy")
        
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f_numpy(x_vals)

        fig = go.Figure()

        # 1. Main Trace with Shading
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, 
            mode='lines',
            name="f(x)",
            line=dict(color='#00d4ff', width=4),
            fill='tozeroy' if calc_type == "Integration" else None,
            fillcolor='rgba(0, 212, 255, 0.2)',
            hovertemplate="<b>x:</b> %{x:.2f}<br><b>y:</b> %{y:.2f}<extra></extra>"
        ))

        # 2. MARK CRITICAL POINTS (Turning Points)
        try:
            derivative = sp.diff(expr, x_sym)
            critical_points = sp.solve(derivative, x_sym)
            real_crit = [float(p) for p in critical_points if p.is_real and -10 <= p <= 10]
            
            if real_crit:
                cp_y = [float(expr.subs(x_sym, p)) for p in real_crit]
                fig.add_trace(go.Scatter(
                    x=real_crit, y=cp_y,
                    mode='markers',
                    name="Turning Points",
                    marker=dict(color='#ff4b4b', size=12, symbol='star', line=dict(width=2, color="white")),
                    text=[f"Peak/Valley: ({p:.2f}, {y:.2f})" for p, y in zip(real_crit, cp_y)],
                    hoverinfo="text"
                ))
        except: pass

        # 3. Elaborative Dark UI
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=18, color="#00d4ff")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='#333', zeroline=True, zerolinecolor='white'),
            yaxis=dict(showgrid=True, gridcolor='#333', zeroline=True, zerolinecolor='white'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=450
        )
        return fig
    except:
        return None

# ==========================================
# 3. INTERFACE TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🧮 Smart Solver", "📸 Photo Math", "📝 Exam Prep"])

# --- TAB 1: SOLVER ---
with tab1:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        calc_type = st.selectbox("I want to:", ["Differentiation", "Integration"], key="tab1_calc")
        user_input = st.text_input("Enter Function:", value="x**3 - 3*x", key="tab1_in")
        
        if st.button("Analyze & Solve", type="primary"):
            try:
                x_sym = sp.Symbol('x')
                expr = sp.sympify(user_input)
                res_sym = sp.diff(expr, x_sym) if calc_type == "Differentiation" else sp.integrate(expr, x_sym)
                
                st.latex(f"Result: {sp.latex(res_sym)}")
                
                with st.spinner("AI thinking..."):
                    prompt = f"Explain the {calc_type} of {user_input} step-by-step for a Class 12 student. The answer is {res_sym}. Use standard LaTeX markers ($$). Also explain what the graph's red stars represent."
                    resp = client.models.generate_content(model=STABLE_MODEL, contents=prompt)
                    st.write(resp.text)
            except Exception as e: st.error(f"Error: {e}")

    with col2:
        fig = generate_elaborate_graph(user_input, f"Interactive View of f(x) = {user_input}", calc_type)
        if fig: st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PHOTO MATH ---
with tab2:
    st.write("Upload your math problem photo below.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="photo_upload")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Problem to Solve", width=400)
        
        if st.button("Visual Analysis & Solve"):
            with st.spinner("Analyzing handwriting..."):
                try:
                    # Request for solution and raw formula for graphing
                    image_prompt = "Solve this step-by-step. Use Markdown and LaTeX. No document headers."
                    formula_prompt = "Return ONLY the raw Python-compatible math formula from this image (e.g. x**2). Nothing else."
                    
                    sol_resp = client.models.generate_content(model=STABLE_MODEL, contents=[img, image_prompt])
                    formula_resp = client.models.generate_content(model=STABLE_MODEL, contents=[img, formula_prompt])
                    
                    col_s, col_g = st.columns([1, 1])
                    with col_s:
                        st.markdown("### Solution")
                        st.write(sol_resp.text)
                    
                    with col_g:
                        f_text = formula_resp.text.strip().replace('`','')
                        fig = generate_elaborate_graph(f_text, f"Visualizing: {f_text}")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Graph not available.")
                except Exception as e: st.error(f"Error: {e}")

# --- TAB 3: EXAM PRACTICE ---
with tab3:
    topic = st.selectbox("Choose Topic", ["Calculus Basics", "Maxima and Minima", "Definite Integrals"])
    
    if st.button("Generate Challenging Question"):
        q_prompt = f"Generate a difficult Class 12 exam question about {topic}. Provide only the question."
        q_resp = client.models.generate_content(model=STABLE_MODEL, contents=q_prompt)
        st.session_state['q_txt'] = q_resp.text
        
        f_prompt = f"Extract just the raw math formula from this: {q_resp.text}. Reply only with the formula like x**2."
        f_resp = client.models.generate_content(model=STABLE_MODEL, contents=f_prompt)
        st.session_state['q_form'] = f_resp.text.strip().replace('`','')
        if 'q_sol' in st.session_state: del st.session_state['q_sol']

    if 'q_txt' in st.session_state:
        c1, c2 = st.columns([1, 1])
        with c1: st.info(f"**Question:**\n\n{st.session_state['q_txt']}")
        with c2:
            fig = generate_elaborate_graph(st.session_state['q_form'], "Problem Visualization")
            if fig: st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Reveal Math Logic"):
            with st.spinner("Processing..."):
                s_prompt = f"Provide a detailed step-by-step LaTeX solution for: {st.session_state['q_txt']}. No document headers. Start with 'Step 1:'."
                s_resp = client.models.generate_content(model=STABLE_MODEL, contents=s_prompt)
                st.session_state['q_sol'] = s_resp.text

    if 'q_sol' in st.session_state:
        st.success("### Step-by-Step Answer")
        st.write(st.session_state['q_sol'])
