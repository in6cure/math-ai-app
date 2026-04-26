import os
import streamlit as st
import sympy as sp
from google import genai

# 1. Page Configuration
st.set_page_config(page_title="Calculus Master AI", page_icon="♾️", layout="centered")
st.title("♾️ Calculus Master AI")
st.write("Solve Differentiation, Integration, and Limits with step-by-step logic.")

# 2. Setup API Client
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key missing!")
    st.stop()
client = genai.Client(api_key=api_key)

# 3. CALCULUS SELECTOR
calc_type = st.selectbox("What do you want to calculate?", 
                         ["Differentiation (Derivative)", "Integration (Anti-derivative)", "Limits"])

user_input = st.text_input("Enter your function (e.g., x**2 + 5*x):", value="x**2 + 5*x")

# Only show limit point if Limits is selected
limit_point = 0
if calc_type == "Limits":
    limit_point = st.number_input("Limit approaches what value?", value=0)

# 4. The "Solve" Button
if st.button("Generate Step-by-Step Solution"):
    with st.spinner("Processing..."):
        try:
            x = sp.Symbol('x')
            expr = sp.sympify(user_input)
            
            # --- MATH LOGIC ---
            if calc_type == "Differentiation (Derivative)":
                result = sp.diff(expr, x)
                task_desc = "differentiate"
            elif calc_type == "Integration (Anti-derivative)":
                result = sp.integrate(expr, x)
                task_desc = "integrate"
            else:
                result = sp.limit(expr, x, limit_point)
                task_desc = f"find the limit as x approaches {limit_point} for"

            # --- AI EXPLAINER ---
            prompt = f"""
            You are a world-class Calculus tutor for Class 12 students.
            The student wants to {task_desc} the function: {user_input}
            The mathematically correct final answer is: {result}

            Provide a clear, pedagogical, step-by-step breakdown using LaTeX.
            Explain the rules used (e.g., Power Rule, Chain Rule, or Fundamental Theorem of Calculus).
            """
            
            response = client.models.generate_content(
                model='gemini-3-flash',
                contents=prompt,
            )
            
            st.success("Completed!")
            st.markdown(f"### Final Answer: ${sp.latex(result)}$")
            st.markdown("---")
            st.markdown("### Step-by-Step Logic")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Error: {e}. Check your math syntax (use * for multiply and ** for power).")
