
# streamlit_app.py
import os
import tempfile

import pandas as pd
import streamlit as st

from pdf_report import create_pdf_report

st.set_page_config(page_title="Soccer Central AI Quality Control", layout="centered")

st.title("Soccer Central AI Quality Control")
st.caption("Upload a survey export (.xlsx) and generate the PDF report.")

survey_type = st.selectbox("Survey type", ["players", "families", "staff"], index=0)

cycle_label = st.text_input("Cycle label (shown in PDF title)", value="Cycle x")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

with st.expander("Debug: show Excel columns and indices", expanded=False):
    if uploaded is not None:
        df_debug = pd.read_excel(uploaded, sheet_name=0)
        for i, c in enumerate(df_debug.columns):
            st.write(f"{i}: {c}")

run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

if run and uploaded is not None:
    with st.spinner("Generating PDF..."):
        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.getbuffer())
            input_path = tmp.name

        # Generate report
        pdf_path = create_pdf_report(
            input_path=input_path,
            cycle_label=cycle_label,
            survey_type=survey_type,
            output_path=None,
        )

        # Offer download
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF report",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
            
st.markdown(
    """
    <style>
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: rgba(49, 51, 63, 0.7);
        background: white;
        border-top: 1px solid rgba(49, 51, 63, 0.15);
        z-index: 999;
    }
    </style>

    <div class="app-footer">
        Program Developed by Rene Barbier for Soccer Central
    </div>
    """,
    unsafe_allow_html=True,
)
