import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from descriptors import calculate_descriptors
from src.ai_agent_new import query_ai_agent
from PIL import Image

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Descriptor Calculator", "AI Agent", "Visualization", "About"])

# --- HOME PAGE ---
if page == "Home":
    import os

    # Path to the banner image
    banner_path = "banner.jpg"

    if os.path.exists(banner_path):
        banner = Image.open(banner_path)
        banner = banner.resize((600, 200))  # Resize banner

        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(banner, use_container_width=False)
    else:
        st.warning("⚠️ Banner image not found. Make sure 'banner.jpg' is in the same folder as app.py!")

    # Main heading + subtitle
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1 style='color: black; font-size: 48px; margin-bottom: 5px;'>QSAR Descriptor Calculator 🧪</h1>
            <h4 style='color: gray; font-weight: normal;'>Explore molecular descriptors and AI-powered insights in one interactive app!</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Page description
    st.markdown("""
    ### Welcome! 👋  

    This app demonstrates **QSAR (Quantitative Structure–Activity Relationship)** analysis with AI-driven insights.

    **Navigation:**
    - 🧮 **Descriptor Calculator** → Upload SMILES and calculate descriptors  
    - 🤖 **AI Agent** → Get AI-powered compound summaries (single or batch mode)  
    - 📈 **Visualization** → Explore plots, correlations, and stats  
    - ℹ️ **About** → Learn more about this project  

    🚀 Designed for learning, research, and fun exploration of molecules!
    """)

# --- DESCRIPTOR CALCULATOR PAGE ---
elif page == "Descriptor Calculator":
    st.header("Descriptor Calculator 🧮")

    uploaded_file = st.file_uploader("📂 Upload CSV file with SMILES", type="csv")

    results = None
    result_df = None

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'SMILES' not in df.columns:
            st.error("CSV must have a column named 'SMILES'")
        else:
            # calculate descriptors (spinner)
            with st.spinner("Calculating descriptors..."):
                results = []
                mol_list = []
                from rdkit.Chem import QED  # RDKit QED

                def _get(desc, candidates):
                    for k in candidates:
                        if k in desc:
                            return desc[k]
                    return None

                for smi in df['SMILES']:
                    desc = calculate_descriptors(smi)
                    mol = Chem.MolFromSmiles(smi)
                    mol_list.append(mol)
                    if desc:
                        # compute QED if mol exists
                        qed_val = None
                        if mol:
                            try:
                                qed_val = float(QED.qed(mol))
                            except Exception:
                                qed_val = None

                        # Lipinski checks
                        mw = _get(desc, ["Molecular Weight", "MolWt", "MW"])
                        logp = _get(desc, ["LogP", "logP", "ALogP"])
                        hbd = _get(desc, ["H-bond Donors", "HBD", "H_donors", "NumHDonors"])
                        hba = _get(desc, ["H-bond Acceptors", "HBA", "H_bond_acceptors", "NumHAcceptors"])

                        def _to_float(x):
                            try:
                                return float(x)
                            except Exception:
                                return None

                        mw_v = _to_float(mw)
                        logp_v = _to_float(logp)
                        hbd_v = _to_float(hbd)
                        hba_v = _to_float(hba)

                        violations = 0
                        if mw_v is not None and mw_v > 500: violations += 1
                        if logp_v is not None and logp_v > 5: violations += 1
                        if hbd_v is not None and hbd_v > 5: violations += 1
                        if hba_v is not None and hba_v > 10: violations += 1

                        desc_copy = desc.copy()
                        desc_copy['SMILES'] = smi
                        desc_copy['QED'] = qed_val
                        desc_copy['Lipinski Violations'] = violations
                        desc_copy['Lipinski Pass'] = (violations == 0)
                        results.append(desc_copy)

            if results:
                result_df = pd.DataFrame(results)
                st.subheader("📊 Descriptor Results")
                st.dataframe(result_df, use_container_width=True)
                st.session_state['results'] = result_df

                csv = result_df.to_csv(index=False)
                st.download_button("💾 Download results as CSV", csv, "qsar_results_with_qed.csv")
            else:
                st.warning("No descriptors could be calculated for the provided SMILES.")

    else:
        st.info("Upload a CSV file to start calculating descriptors.")

    # --- ANALYSIS OPTIONS (always visible) ---
    st.markdown("---")
    st.markdown("### 🔎 Quick Analyses & Visualization")

    c1, c2, c3 = st.columns(3)
    with c1:
        show_corr = st.checkbox("Correlation heatmap", value=True)
    with c2:
        show_pca_2d = st.checkbox("PCA 2D", value=False)
    with c3:
        show_pca_3d = st.checkbox("PCA 3D", value=False)

    q1, q2 = st.columns(2)
    with q1:
        show_qed = st.checkbox("Show QED summary", value=True)
    with q2:
        show_lipinski = st.checkbox("Show Lipinski summary", value=True)

    # prepare numeric data if results exist
    if result_df is not None:
        numeric_df = result_df.select_dtypes(include='number').copy()
        n_numeric = numeric_df.shape[1]
    else:
        numeric_df = None
        n_numeric = 0

    try:
        import plotly.express as px
    except Exception:
        st.error("Plotly not installed. Run `pip install plotly`")
        px = None

    # --- Correlation heatmap ---
    if show_corr:
        st.subheader("🔥 Descriptor Correlation Heatmap")
        if result_df is None:
            st.info("Upload and calculate descriptors to see correlation heatmap.")
        elif n_numeric < 2 or px is None:
            st.info("Need at least 2 numeric descriptor columns.")
        else:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)

    # --- PCA (2D / 3D) ---
    if (show_pca_2d or show_pca_3d):
        try:
            from sklearn.decomposition import PCA
        except Exception:
            st.error("scikit-learn not installed. Run `pip install scikit-learn`")
            PCA = None

        if result_df is None or PCA is None:
            st.info("Upload data to run PCA.")
        else:
            pca_input = numeric_df.dropna(axis=0, how='any')
            if pca_input.shape[0] < 2:
                st.info("Not enough complete rows for PCA.")
            else:
                if show_pca_2d:
                    st.subheader("📐 PCA 2D")
                    pca2 = PCA(n_components=2)
                    comps2 = pca2.fit_transform(pca_input.values)
                    pca_df2 = pd.DataFrame(comps2, columns=['PC1','PC2'])
                    pca_df2['SMILES'] = result_df.loc[pca_input.index, 'SMILES'].values
                    fig2 = px.scatter(pca_df2, x='PC1', y='PC2', hover_data=['SMILES'], title="PCA (2 components)")
                    st.plotly_chart(fig2, use_container_width=True)

                if show_pca_3d:
                    st.subheader("📐 PCA 3D")
                    if pca_input.shape[0] < 3:
                        st.info("Need at least 3 molecules for 3D PCA plot.")
                    else:
                        pca3 = PCA(n_components=3)
                        comps3 = pca3.fit_transform(pca_input.values)
                        pca_df3 = pd.DataFrame(comps3, columns=['PC1','PC2','PC3'])
                        pca_df3['SMILES'] = result_df.loc[pca_input.index, 'SMILES'].values
                        try:
                            fig3 = px.scatter_3d(pca_df3, x='PC1', y='PC2', z='PC3',
                                                 hover_data=['SMILES'], title="PCA (3 components)")
                            st.plotly_chart(fig3, use_container_width=True)
                        except Exception:
                            st.info("Plotly 3D not available.")

    # --- QED summary ---
    if show_qed:
        st.subheader("💊 QED (Drug-likeness)")
        if result_df is None:
            st.info("Upload and calculate descriptors to see QED summary.")
        elif 'QED' not in result_df.columns:
            st.info("QED not found in results.")
        else:
            if px is None:
                st.write(result_df[['SMILES','QED']].sort_values('QED', ascending=False).head(10))
            else:
                fig_q = px.histogram(result_df, x='QED', nbins=20, title="QED distribution")
                st.plotly_chart(fig_q, use_container_width=True)
                st.table(result_df[['SMILES','QED']].sort_values('QED', ascending=False).head(10))

    # --- Lipinski summary ---
    if show_lipinski:
        st.subheader("🧾 Lipinski Rule-of-5 Summary")
        st.markdown("Rules: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10")
        if result_df is None:
            st.info("Upload and calculate descriptors to see Lipinski summary.")
        elif 'Lipinski Violations' in result_df.columns:
            counts = result_df['Lipinski Violations'].value_counts().sort_index()
            st.write("Violation counts:", counts.to_dict())
            st.table(result_df[['SMILES','Lipinski Violations']].sort_values('Lipinski Violations').head(10))
        else:
            st.info("Lipinski info not available.")

# --- AI AGENT PAGE ---
elif page == "AI Agent":
    st.title("🤖 AI Agent – Know Your Compound Smarter")
    st.markdown("""
    Welcome to the **AI Molecular Assistant**!  
    You can either enter a **single SMILES** or **upload a CSV file** of compounds.  
    The AI will provide:  
    - 🧾 A summary of the compound(s)  
    - 🧪 Insights on drug-likeness & solubility  
    - ⚠️ Warnings and limitations  
    - 🌍 Potential applications
    """)

    option = st.radio("Choose input mode:", ["Enter a single SMILES", "Upload a CSV file"])

    from io import BytesIO
    from PIL import Image

    # Function to generate molecule thumbnail
    def mol_image(smiles, size=(100, 100)):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=size)
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf
        return None

    # -------------------- SINGLE SMILES --------------------
    if option == "Enter a single SMILES":
        smiles_input = st.text_input("✍️ Enter SMILES of your compound")
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                # Two-column layout
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("🖼️ Molecule Structure")
                    st.image(Draw.MolToImage(mol, size=(300, 300)))

                with col2:
                    st.subheader("📊 Calculated Descriptors")
                    desc = calculate_descriptors(smiles_input)
                    if desc:
                        desc_df = pd.DataFrame([desc])
                        st.table(desc_df)

                # AI Analysis below columns
                if desc:
                    st.subheader("🤖 AI Agent Summary")
                    if st.button("🔍 Analyze Compound with AI"):
                        with st.spinner("AI is analyzing your compound..."):
                            prompt = (
                                f"Here are the QSAR molecular descriptors for a compound:\n\n{desc_df.to_string(index=False)}\n\n"
                                f"Please provide a detailed but concise summary in a structured format:\n"
                                f"- General compound description\n"
                                f"- Possible drug-likeness\n"
                                f"- Solubility and lipophilicity\n"
                                f"- Warnings or limitations\n"
                                f"- Potential real-world applications\n"
                            )
                            try:
                                ai_response = query_ai_agent(prompt)
                                st.success("✅ AI Analysis Complete")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            else:
                st.error("Invalid SMILES! Please enter a valid molecule.")

    # -------------------- BATCH CSV UPLOAD --------------------
    elif option == "Upload a CSV file":
        uploaded_file = st.file_uploader("📂 Upload CSV file with a 'SMILES' column", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "SMILES" not in df.columns:
                st.error("CSV must have a column named 'SMILES'")
            else:
                st.success("✅ File uploaded successfully")
                st.subheader("📊 Preview of Uploaded Data")
                st.write(df.head())

                # Calculate descriptors and generate thumbnails
                results = []
                structures = []
                for smi in df["SMILES"]:
                    desc = calculate_descriptors(smi)
                    if desc:
                        desc["SMILES"] = smi
                        results.append(desc)
                        structures.append(mol_image(smi, size=(100, 100)))

                if results:
                    result_df = pd.DataFrame(results)
                    result_df["Structure"] = structures

                    # Display each molecule with thumbnail + descriptors
                    st.subheader("📊 Calculated Descriptors with Structures")
                    for i in range(len(result_df)):
                        st.markdown(f"**SMILES:** {result_df.loc[i,'SMILES']}")
                        st.image(result_df.loc[i,'Structure'], width=100)
                        st.write(result_df.loc[i].drop("Structure"))

                    # AI Batch Analysis
                    st.subheader("🤖 AI Agent Batch Analysis")
                    if st.button("🔍 Analyze All Compounds with AI"):
                        with st.spinner("AI is analyzing your compounds..."):
                            result_text = result_df.head(10).drop(columns="Structure").to_string(index=False)
                            prompt = (
                                f"Analyze the following QSAR molecular descriptors for multiple compounds:\n\n{result_text}\n\n"
                                f"Provide insights:\n"
                                f"- Which molecules look more drug-like\n"
                                f"- Solubility/lipophilicity trends\n"
                                f"- Any outliers or warnings\n"
                                f"- General summary in table + text"
                            )
                            try:
                                ai_response = query_ai_agent(prompt)
                                st.success("✅ AI Batch Analysis Complete")
                                st.write(ai_response)
                            except Exception as e:
                                st.error(f"❌ Error: {e}")

# --- VISUALIZATION PAGE ---
elif page == "Visualization":
    st.title("📈 Visualization of Descriptors")

    if "results" in st.session_state:
        result_df = st.session_state['results']

        import plotly.express as px

        # Interactive histograms for each descriptor
        st.subheader("📊 Descriptor Distributions (Interactive)")
        for col in result_df.columns:
            if col != "SMILES":
                fig = px.histogram(
                    result_df,
                    x=col,
                    nbins=20,
                    title=f"Distribution of {col}",
                    labels={col: col},
                    color_discrete_sequence=["skyblue"]
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap (interactive)
        st.subheader("🔥 Correlation Heatmap")
        corr = result_df.drop(columns=["SMILES"]).corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Descriptors"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("📑 Summary Statistics")
        st.dataframe(result_df.drop(columns=["SMILES"]).describe())

    else:
        st.warning("⚠️ No results found! Please calculate descriptors first in the *Descriptor Calculator* page.")

# --- ABOUT PAGE ---
elif page == "About":
    st.title("ℹ️ About this App")
    st.markdown("""
    This project demonstrates how **QSAR (Quantitative Structure–Activity Relationship)** descriptors  
    can be calculated and visualized using **RDKit** and **Streamlit**.  

    ### Features:
    - 🧮 Descriptor calculation from SMILES  
    - 📈 Visualizations (histograms, correlations, stats)  
    - 🤖 AI-powered compound insights (single or batch mode)  
    - 💾 Download results for further analysis  

    **Developed for learning purposes 🚀**
    """)
