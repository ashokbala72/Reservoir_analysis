import os
import streamlit as st
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai_cache = {}
metric_change_cache = {}

st.set_page_config(page_title="📂 Reservoir GenAI Assistant", layout="wide")

sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
LOCATIONS = [
    "ghawar_field", "mumbai_high", "permian_basin", "burgan_field", "south_pars_north_dome",
    "krishna_godavari_basin", "bohai_bay", "browse_basin", "taranaki_basin", "brent_field"
]

drill_zones = ['Zone A', 'Zone B', 'Zone C']
if "selected_location" not in st.session_state:
    st.session_state.selected_location = LOCATIONS[0]
if "drill_zone" not in st.session_state:
    st.session_state.drill_zone = drill_zones[0]

# Tabs

emoji_titles = [
    "🏠 Overview",
    "📍 Location",
    "🧪 ECLIPSE",
    "🧪 CMG",
    "🧪 tNavigator",
    "🧪 OPM",
    "🧪 MRST",
    "📊 Metrics for Engineers",
    "📈 Summary",
    "🧠 Recommendations",
    "🎯 Drill Targeting",
    "🔍 Logs & Petrophysics",
    "🗺️ Seismic Viewer",
    "📐 Volumetrics Estimation",
    "🧬 Rock Typing",
    "📚 Ask a Question",
]
tabs = st.tabs(emoji_titles)


# 🏠 Overview
with tabs[0]:
    st.title("🛢️ Exploration and Reservoir Characterization")
    st.markdown("""
## 🧠 Reservoir Exploration Assistant — Step-by-Step in Simple Terms
### *(Now with how you can replace simulated/mock data with real inputs)*

### 🏠 1. Overview
> 📘 *Your user manual for the tool.*

- Helps engineers explore oil fields and pick drilling spots using AI + simulator data.
- Uses:
  - ✅ Real simulators: OPM, MRST
  - 🔄 Mock simulators: ECLIPSE, CMG, tNavigator

🔧 To use real data: connect file readers or APIs from those tools (UNRST, INIT, CMOST, RESQML, etc.)

### 📍 2. Location
Pick location & drill zone. Refresh to simulate/fetch updated results.

### X, Y, Z Coordinates
- X: East–West
- Y: North–South
- Z: Depth (negative = underground)
🧭 Like GPS for drilling.

### 🧪 3–7. Simulators
Each simulator tab shows oil rate, pressure, and water cut trends with AI commentary.

🔧 To make them real: integrate CSVs or APIs from simulators like Petrel, Eclipse, CMG.

### 📊 8. Metrics
Compares stability (variation) across simulators. Picks most reliable one.

### 📈 9. Summary
One graph comparing all platforms. GenAI explains what it sees.

### 🧠 10. Recommendations
AI gives 3 field actions — what to do, why it helps, expected impact.

### 🎯 11. Drill Targeting
Shows best coordinates to drill for each simulator + 3D view.

### 🔍 12. Logs
Shows PHI, Sw, NetPay by depth. AI tells you where the good oil is.
🔧 Plug in LAS/DLIS files for real logs.

### 🗺️ 13. Seismic Viewer
Shows synthetic seismic image + GenAI interpretation.
🔧 Upload SEG-Y to replace synthetic.

### 📐 14. Volumetrics
Estimate how much oil is in place. AI explains the estimate.

### 🧬 15. Rock Typing
Clusters rock zones from logs. GenAI interprets clusters.
🔧 Use core data/facies models for accuracy.

### 📚 16. Ask a Question
Ask GenAI anything. It answers based on the location, logs, XYZ and best simulator.

---

This tool helps you make confident exploration decisions without needing to be a simulation expert.
""")
with tabs[1]:
    st.header("📍 Select Location")
    st.session_state.selected_location = st.selectbox("Choose a location", LOCATIONS, index=LOCATIONS.index(st.session_state.selected_location))
    st.session_state.drill_zone = st.selectbox('Select Drill Zone', drill_zones, index=drill_zones.index(st.session_state.drill_zone))
    if st.button("🔄 Refresh Now", key="_refresh_now_button"):
        location = st.session_state.selected_location
        folder = f"sim_data/{location}"
        os.makedirs(folder, exist_ok=True)

        location_seed = abs(hash(location)) % 10000
        for sim in sim_names:
            base = location_seed % 300 + 700
            df = pd.DataFrame({
                "TIME_DAYS": [0, 30, 60, 90, 120],
                "OIL_RATE": [random.randint(base, base + 100) for _ in range(5)],
                "PRESSURE": [random.randint(2500 + base % 300, 3500 + base % 300) for _ in range(5)],
                "WATER_CUT": [random.randint(5, 25) for _ in range(5)],
                "X": [random.uniform(1000, 2000) for _ in range(5)],
                "Y": [random.uniform(500, 1000) for _ in range(5)],
                "Z": [random.uniform(-3000, -2000) for _ in range(5)],
                "UPDATED_AT": [int(time.time())] * 5
            })
            df.to_csv(f"{folder}/{sim}_output.csv", index=False)
        st.success(f"✅ Refresh completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Simulator Tabs
for idx, sim in enumerate(sim_names, start=2):
    with tabs[idx]:
        st.header(f"🧪 {sim.upper()} Simulator")
        if st.button(f"🔄 Refresh {sim.upper()} Data"):
            location = st.session_state.selected_location
            folder = f"sim_data/{location}"
            os.makedirs(folder, exist_ok=True)
            base = abs(hash(location + sim)) % 300 + 700
            df = pd.DataFrame({
                "TIME_DAYS": [0, 30, 60, 90, 120],
                "OIL_RATE": [random.randint(base, base + 100) for _ in range(5)],
                "PRESSURE": [random.randint(2500 + base % 300, 3500 + base % 300) for _ in range(5)],
                "WATER_CUT": [random.randint(5, 25) for _ in range(5)],
                "X": [random.uniform(1000, 2000) for _ in range(5)],
                "Y": [random.uniform(500, 1000) for _ in range(5)],
                "Z": [random.uniform(-3000, -2000) for _ in range(5)],
                "UPDATED_AT": [int(time.time())] * 5
            })
            df.to_csv(f"{folder}/{sim}_output.csv", index=False)
            st.success(f"✅ {sim.upper()} Data Refreshed")

        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            for metric in ["OIL_RATE", "PRESSURE", "WATER_CUT"]:
                label = metric
                if sim in metric_change_cache and metric in metric_change_cache[sim]:
                    old_val = metric_change_cache[sim][metric]
                    new_val = df[metric].iloc[-1]
                    direction = "⬆️" if new_val > old_val else ("⬇️" if new_val < old_val else "➡️")
                    label += f" {direction}"
                ax.plot(df["TIME_DAYS"], df[metric], label=label)
            ax.set_title(f"{sim.upper()} Simulation Trends")
            ax.legend()
            st.pyplot(fig)

            if sim not in metric_change_cache:
                metric_change_cache[sim] = {}
            for metric in ["OIL_RATE", "PRESSURE", "WATER_CUT"]:
                metric_change_cache[sim][metric] = df[metric].iloc[-1]

            cache_key = f"{st.session_state.selected_location}_{sim}"
            if cache_key not in genai_cache:
                prompt = f"You are a petroleum engineer. Summarize trends in oil rate, pressure, and water cut from this {sim} simulation at {st.session_state.selected_location} ({st.session_state.drill_zone}).\n\n{df.head().to_csv(index=False)}"
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                    )
                    genai_cache[cache_key] = response.choices[0].message.content
                except Exception as e:
                                st.warning(f"GenAI error: {str(e)}")
            st.markdown("#### 💡 GenAI Suggestions")
            st.info(genai_cache.get(cache_key, "No data available."))
        else:
            st.warning(f"❌ CSV not found for {sim.upper()} at {st.session_state.selected_location}")

# 📈 Summary
with tabs[len(sim_names) + 3]:
    st.header("📈 Cross-Simulator Summary")
    selected_metric = st.selectbox("Choose metric for trend analysis", ["OIL_RATE", "PRESSURE", "WATER_CUT"])
    dfs = []
    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['SIMULATOR'] = sim.upper()
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs)
        fig, ax = plt.subplots(figsize=(6, 3))
        for sim in sim_names:
            df_sub = combined_df[combined_df['SIMULATOR'] == sim.upper()]
            ax.plot(df_sub["TIME_DAYS"], df_sub[selected_metric], label=f"{sim.upper()} - {selected_metric}")
        ax.set_title(f"{selected_metric} Trends Across Simulators")
        ax.legend()
        st.pyplot(fig)

        cache_key = f"summary_{selected_metric}_{st.session_state.selected_location}"
        if cache_key not in genai_cache:
            csv_data = combined_df[["TIME_DAYS", "SIMULATOR", selected_metric]].head().to_csv(index=False)
            prompt = f"You are a reservoir analyst. Provide a short summary of trends in {selected_metric.lower()} across all simulators at {st.session_state.selected_location} ({st.session_state.drill_zone}).\n\n{csv_data}"
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                genai_cache[cache_key] = response.choices[0].message.content
            except Exception as e:
                            st.warning(f"GenAI error: {str(e)}")
        st.markdown("#### 📌 GenAI Summary")
        st.info(genai_cache.get(cache_key, "No summary available."))





# 📊 Metrics for Engineers
with tabs[len(sim_names) + 2]:
    st.header("📊 Engineering Metrics & Platform Reliability")

    metric_data = []

    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            oil_avg = df["OIL_RATE"].mean()
            oil_var = df["OIL_RATE"].var()
            pres_avg = df["PRESSURE"].mean()
            pres_var = df["PRESSURE"].var()

            stability_score = round((oil_var + pres_var) / 2, 2)

            metric_data.append({
                "Simulator": sim.upper(),
                "Avg Oil Rate": round(oil_avg, 2),
                "Avg Pressure": round(pres_avg, 2),
                "Stability Score (lower is better)": stability_score
            })

    if metric_data:
        df_metric = pd.DataFrame(metric_data)
        df_metric = df_metric.sort_values("Stability Score (lower is better)")
        st.dataframe(df_metric, use_container_width=True)

        best_sim = df_metric.iloc[0]["Simulator"]
        st.success(f"🏆 **Most Reliable Simulator:** {best_sim} (based on stability of Oil Rate and Pressure)")
    else:
        st.warning("No simulation data available for analysis.")

# 🧠 Recommendations
with tabs[len(sim_names) + 4]:
    st.header("🧠 GenAI Operational Recommendations")
    rec_focus = st.radio("Focus of Recommendations", ["Production Optimization", "Water Management", "Pressure Maintenance"], index=0)

    latest_metrics = {"oil_rate": [], "pressure": [], "water_cut": []}
    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            latest_metrics["oil_rate"].append(df["OIL_RATE"].iloc[-1])
            latest_metrics["pressure"].append(df["PRESSURE"].iloc[-1])
            latest_metrics["water_cut"].append(df["WATER_CUT"].iloc[-1])

    if all(latest_metrics.values()):
        avg_oil = round(sum(latest_metrics["oil_rate"]) / len(latest_metrics["oil_rate"]), 2)
        avg_pres = round(sum(latest_metrics["pressure"]) / len(latest_metrics["pressure"]), 2)
        avg_wc = round(sum(latest_metrics["water_cut"]) / len(latest_metrics["water_cut"]), 2)
    else:
        avg_oil, avg_pres, avg_wc = "N/A", "N/A", "N/A"

    cache_key = f"recommend_{st.session_state.selected_location}_{rec_focus}"
    if st.button("🔄 Regenerate Recommendations", key="_regenerate_recommendations_button") or cache_key not in genai_cache:
        prompt = f"""
        You are a petroleum production expert.
        Based on simulator trends at {st.session_state.selected_location} ({st.session_state.drill_zone}):
        - Average Oil Rate: {avg_oil}
        - Average Pressure: {avg_pres}
        - Average Water Cut: {avg_wc}

        Provide 3 concrete recommendations focused on {rec_focus}.
        Format:
        - ✅ Action:
        - 🔍 Why:
        - 📈 Expected Impact:
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            genai_cache[cache_key] = response.choices[0].message.content
        except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")

    st.markdown("#### 🛠️ Targeted Recommendations")
    st.success(genai_cache.get(cache_key, "No recommendations available."))





# 🎯 Drill Targeting
with tabs[len(sim_names) + 5]:
    st.header("🎯 Targeted Drilling Coordinates")

    st.markdown("""
    ### 🧭 Coordinate Explanation
    - **X**: East-West coordinate in meters (horizontal)
    - **Y**: North-South coordinate in meters (horizontal)
    - **Z**: Depth below surface (vertical; typically negative)
    """)
    st.markdown("We analyze simulation data to suggest the most promising XYZ coordinates for drilling.")

    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if {'X', 'Y', 'Z', 'OIL_RATE', 'WATER_CUT'}.issubset(df.columns):
                df["SCORE"] = df["OIL_RATE"] / (df["WATER_CUT"] + 1)
                best_point = df.loc[df["SCORE"].idxmax()]
                st.markdown(f"##### 🔍 {sim.upper()}")
                x = round(best_point['X'], 2)
                y = round(best_point['Y'], 2)
                z = round(best_point['Z'], 2)
                score = round(best_point['SCORE'], 2)
                st.markdown("**Recommended Drill Location Parameters:**")
                st.markdown(f"""
| Parameter          | Value         | Meaning                                                              |
|--------------------|---------------|----------------------------------------------------------------------|
| **X**              | `{x} meters`  | East-West horizontal coordinate in the reservoir model               |
| **Y**              | `{y} meters`  | North-South horizontal coordinate                                     |
| **Z**              | `{z} meters`  | Depth below surface (typically negative)                              |
| **Composite Score**| `{score}`     | Index based on oil rate and water cut — higher is better             |
                """)
                st.caption("📌 Composite Score = OIL_RATE / (WATER_CUT + 1)")

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df["X"], df["Y"], df["Z"], c=df["SCORE"], cmap='viridis', s=60)
                ax.set_title(f"Drill Zone Intensity - {sim.upper()}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                st.pyplot(fig)

    # 📊 Drill Targeting Comparison Across Simulators
    comparison_data = []

    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if {'X', 'Y', 'Z', 'OIL_RATE', 'WATER_CUT'}.issubset(df.columns):
                df["SCORE"] = df["OIL_RATE"] / (df["WATER_CUT"] + 1)
                best = df.loc[df["SCORE"].idxmax()]
                comparison_data.append({
                    "Simulator": sim.upper(),
                    "X": round(best["X"], 2),
                    "Y": round(best["Y"], 2),
                    "Z": round(best["Z"], 2),
                    "Composite Score": round(best["SCORE"], 2)
                })

    if comparison_data:
        st.markdown("### 📊 Cross-Simulator Drill Coordinate Comparison")
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values("Composite Score", ascending=False)
        st.dataframe(comp_df, use_container_width=True)

        best_row = comp_df.iloc[0]
        st.success(f"💡 Most probable optimal drilling point is in **{best_row['Simulator']}** at X: {best_row['X']}, Y: {best_row['Y']}, Z: {best_row['Z']} with a composite score of {best_row['Composite Score']}.")


# === Merged Exploration Tabs ===




with tabs[11]:
    st.header("🔍 Logs & Petrophysics")

    fallback_sim = st.session_state.get("best_simulator", "eclipse").lower()
    fallback_path = f"sim_data/{st.session_state.selected_location}/{fallback_sim}_output.csv"

    df = None
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        st.info(f"ℹ️ Loaded log data from {fallback_sim.upper()} simulator output")

    if df is not None:
        if 'DEPTH' not in df.columns:
            df['DEPTH'] = df.index * 10 + 1000  # synthetic depth range
        if 'NPHI' not in df.columns:
            df['NPHI'] = [random.uniform(0.08, 0.16) for _ in range(len(df))]
        if 'RHOB' not in df.columns:
            df['RHOB'] = [random.uniform(2.2, 2.5) for _ in range(len(df))]
        if 'RT' not in df.columns:
            df['RT'] = [random.uniform(5, 20) for _ in range(len(df))]

        df['PHI'] = df['NPHI']
        df['Sw'] = 0.62 / (df['RT'] ** 0.25)
        df['NetPay'] = np.where((df['PHI'] > 0.1) & (df['Sw'] < 0.5), 1, 0)

        st.dataframe(df[['DEPTH', 'PHI', 'Sw', 'NetPay']].head(), use_container_width=True)
        st.line_chart(df.set_index('DEPTH')[['PHI', 'Sw']])
        productive_depths = df[df['NetPay'] == 1]['DEPTH']
        if not productive_depths.empty:
            st.success(f"✅ Productive zone range: {productive_depths.min()} - {productive_depths.max()} meters")
        # GenAI summary for logs
        summary_df = df[['DEPTH', 'PHI', 'Sw', 'NetPay']].copy()
        summary_csv = summary_df.head(15).to_csv(index=False)
        prompt = f"""
        You are a petrophysicist analyzing well log data to evaluate reservoir quality.

        - Location: {st.session_state.selected_location}
        - Zone: {st.session_state.drill_zone}

        Provide a technical interpretation of the PHI, Sw, and NetPay values shown below.
        Indicate likely pay zones, reservoir quality, and hydrocarbon saturation patterns.

        {summary_csv}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            st.markdown("### 🧠 GenAI Interpretation")
            st.info(response.choices[0].message.content)
        except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")

        else:
            st.warning("No productive intervals found.")
    else:
        st.warning("⚠️ No data found to display logs.")


with tabs[12]:
    st.header("🗺️ Seismic Viewer")

    

    inline = st.number_input("Inline", min_value=0, max_value=1000, value=245)
    crossline = st.number_input("Crossline", min_value=0, max_value=1000, value=400)
    fig, ax = plt.subplots(figsize=(6, 4))
    np.random.seed(inline + crossline)
    seismic_data = np.random.normal(0, 1, (100, 100)) + np.sin(np.linspace(0, 10, 100))[None, :]
    ax.imshow(seismic_data, cmap="gray", aspect="auto")
    ax.set_title(f"Synthetic Seismic Inline - {inline}, Crossline - {crossline}")
    st.pyplot(fig)


    if st.button("🧠 Interpret Seismic at Inline/Crossline", key="_interpret_seismic_at_inline_crossline_button"):
        prompt = f"""
        You are a geophysicist analyzing a synthetic seismic inline slice for a potential drilling site.

        - Location: {st.session_state.selected_location.replace('_', ' ').title()}
        - Zone: {st.session_state.drill_zone}
        - Inline: {inline}
        - Crossline: {crossline}
        - Purpose: Assess hydrocarbon potential and identify structural features relevant for drilling.

        Interpret the inline seismic data and describe:
        1. Any indicators of structural traps or closures (e.g., anticlines, domes)
        2. Fault lines, pinch-outs, or stratigraphic traps
        3. Depth implications and potential drilling risks
        4. Likelihood of hydrocarbons at this location

        Keep the language technical but clear for petroleum engineers.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            st.markdown("### 🧠 GenAI Interpretation")
            st.info(response.choices[0].message.content)
        except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")


with tabs[13]:
    st.header("📐 Field Volumetrics Calculator")
    A = st.number_input("Area (acres)", value=1200.0)
    h = st.number_input("Net Pay Thickness (feet)", value=50.0)
    phi = st.number_input("Porosity (φ)", value=0.12)
    Sw = st.number_input("Water Saturation (Sw)", value=0.3)
    Boi = st.number_input("Formation Volume Factor (Boi)", value=1.2)
    OOIP = 7758 * A * h * phi * (1 - Sw) / Boi
    st.success(f"🛢️ OOIP Estimate: {OOIP:,.0f} barrels")

    if st.button("🧠 Summarize OOIP Estimate", key="_summarize_ooip_estimate_button"):
        prompt = f"""You are a reservoir engineer. Summarize OOIP volumetrics for:
- Location: {st.session_state.selected_location}
- Zone: {st.session_state.drill_zone}
- Area: {A} acres
- Net Pay: {h} ft
- Porosity: {phi}
- Sw: {Sw}
- Boi: {Boi}
"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            st.markdown("### 🧠 GenAI Summary")
            st.info(response.choices[0].message.content)
        except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")


with tabs[14]:
    st.header("🧬 Rock Typing & Facies Clustering")

    if "logs_df" not in st.session_state:
        fallback_sim = st.session_state.get("best_simulator", "eclipse").lower()
        path = f"sim_data/{st.session_state.selected_location}/{fallback_sim}_output.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['DEPTH'] = df.index * 10 + 1000
            df['NPHI'] = df.get('NPHI', pd.Series([0.12]*len(df)))
            df['RHOB'] = df.get('RHOB', pd.Series([2.4]*len(df)))
            df['GR'] = df.get('GR', pd.Series([80]*len(df)))
            st.session_state.logs_df = df

    if "logs_df" in st.session_state:
        df = st.session_state.logs_df.copy()
        features = st.multiselect("Select Log Features", ['NPHI', 'RHOB', 'GR'], default=['NPHI', 'RHOB'])
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=3)
        model = KMeans(n_clusters=n_clusters)
        df['Cluster'] = model.fit_predict(df[features])
        pca = PCA(n_components=2)
        pca_vals = pca.fit_transform(df[features])
        df['PC1'], df['PC2'] = pca_vals[:, 0], pca_vals[:, 1]
        sns_fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax)
        st.pyplot(sns_fig)
        st.dataframe(df.groupby('Cluster')[features].mean())
        if not df.empty:
            cluster_summary = df.groupby('Cluster')[features].mean().reset_index()
            prompt = f"""
            You are a reservoir geologist evaluating facies clusters from well log data.

            - Location: {st.session_state.selected_location}
            - Zone: {st.session_state.drill_zone}
            - Clusters: {n_clusters}
            - Features used: {', '.join(features)}

            Provide a concise interpretation of each cluster in terms of potential lithology, porosity trends, and depositional characteristics.
            Use the cluster averages below to guide your analysis:

            {cluster_summary.to_string(index=False)}
            """
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                st.markdown("### 🧠 GenAI Rock Typing Summary")
                st.info(response.choices[0].message.content)
            except Exception as e:
                            st.warning(f"GenAI error: {str(e)}")

    else:
        st.warning("No valid logs data found for clustering.")




    if st.button("Run GenAI", key="run_genai_button_extra_0"):
        try:
            location = st.session_state.selected_location
            zone = st.session_state.drill_zone

            best_sim = "eclipse"
            best_coords = ""
            for sim in sim_names:
                path = f"sim_data/{location}/{sim}_output.csv"
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if {'X', 'Y', 'Z', 'OIL_RATE', 'WATER_CUT'}.issubset(df.columns):
                        df["SCORE"] = df["OIL_RATE"] / (df["WATER_CUT"] + 1)
                        best_point = df.loc[df["SCORE"].idxmax()]
                        best_coords = f"X: {round(best_point['X'], 2)}, Y: {round(best_point['Y'], 2)}, Z: {round(best_point['Z'], 2)}"
                        best_sim = sim
                        break

            system_context = f"""
You are a reservoir exploration assistant.

- Location: {location.replace('_', ' ').title()}
- Zone: {zone}
- Most promising simulator: {best_sim.upper()}
- Suggested drill coordinates: {best_coords}

The user may ask anything related to this context. Use all available internal information to answer. Avoid vague or generic advice.
"""

            full_q = f"{system_context}\n\nUser question: {user_q}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_q}],
                temperature=0.4,
            )
            st.success(response.choices[0].message.content)
        except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")


with tabs[15]:
    st.header("📚 Ask a Question")
    user_q = st.text_area("Ask about this zone")

    if st.button("Run GenAI", key="run_genai_button_extra_1"):
        try:
            location = st.session_state.selected_location
            zone = st.session_state.drill_zone

            best_sim = "eclipse"
            best_coords = ""
            for sim in sim_names:
                path = f"sim_data/{location}/{sim}_output.csv"
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if {'X', 'Y', 'Z', 'OIL_RATE', 'WATER_CUT'}.issubset(df.columns):
                        df["SCORE"] = df["OIL_RATE"] / (df["WATER_CUT"] + 1)
                        best_point = df.loc[df["SCORE"].idxmax()]
                        best_coords = f"X: {round(best_point['X'], 2)}, Y: {round(best_point['Y'], 2)}, Z: {round(best_point['Z'], 2)}"
                        best_sim = sim
                        break

            system_context = f"""
You are a reservoir exploration assistant.

- Location: {location.replace('_', ' ').title()}
- Zone: {zone}
- Most promising simulator: {best_sim.upper()}
- Suggested drill coordinates: {best_coords}

The user may ask anything related to this context. Use all available internal information to answer. Avoid vague or generic advice.
"""

            full_q = f"{system_context}\n\nUser question: {user_q}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_q}],
                temperature=0.4,
            )
            st.success(response.choices[0].message.content)
        except Exception as e:
            st.warning(f"GenAI error: {str(e)}")
