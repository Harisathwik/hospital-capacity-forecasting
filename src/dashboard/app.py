"""
MLOps Monitoring Dashboard for ICU Occupancy Forecasting.
Displays data health, feature drift, and model status from JSON reports.
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# --- Config ---
REPORTS_DIR = Path("data/reports")
MODEL_VERSION_FILE = Path("data/processed/model_version.txt") # Hypothetical or derived from MLflow

st.set_page_config(
    page_title="ICU Forecast Monitoring",
    page_icon="🏥",
    layout="wide"
)

def load_latest_report():
    """Load the most recent monitoring report JSON."""
    reports = sorted(REPORTS_DIR.glob("monitoring_report_*.json"), reverse=True)
    if not reports:
        return None
    with open(reports[0], "r") as f:
        return json.load(f)

# --- UI Header ---
st.title("🏥 ICU Bed Demand Monitoring")
st.markdown("Real-time drift detection and data health for hospital staffing forecasts.")

report = load_latest_report()

if report is None:
    st.error("No monitoring reports found in `data/reports/`. Please run the drift pipeline first.")
    st.stop()

# --- Sidebar: Global Status ---
with st.sidebar:
    st.header("System Status")
    generated_at = report.get("generated_at", "Unknown")
    st.info(f"**Last Report:**\n{generated_at}")
    
    overall_drift = report.get("drift_report", {}).get("overall_drift", "unknown")
    health = report.get("health_report", {}).get("overall_health", "unknown")
    
    # Color status
    drift_color = "red" if "significant" in overall_drift else "orange" if "moderate" in overall_drift else "green"
    health_color = "green" if health == "healthy" else "red"
    
    st.markdown(f"**Drift Status:** :{drift_color}[{overall_drift.upper()}]")
    st.markdown(f"**Data Health:** :{health_color}[{health.upper()}]")
    
    if st.button("Refresh Dashboard"):
        st.rerun()

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Drift Analysis", "🛡️ Data Health", "🚨 Alert Log"])

with tab1:
    st.subheader("Feature Distribution Drift")
    
    drift_results = report.get("drift_report", {}).get("per_feature_drift", [])
    if not drift_results:
        st.warning("No drift data available.")
    else:
        df_drift = pd.DataFrame(drift_results)
        # Clean data: remove errors
        df_drift = df_drift[df_drift["error"].isna() if "error" in df_drift.columns else slice(None)]
        
        # PSI Bar Chart
        fig = px.bar(
            df_drift, 
            x="feature", 
            y="psi", 
            color="drift_level",
            color_discrete_map={"none": "green", "moderate": "orange", "significant": "red"},
            title="Population Stability Index (PSI) per Feature",
            labels={"psi": "PSI Value", "feature": "Feature"}
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="gray", annotation_text="Moderate")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Significant")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_drift, use_container_width=True)

with tab2:
    st.subheader("Data Quality Metrics")
    
    health = report.get("health_report", {})
    checks = health.get("checks", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Summary metrics
    metrics = [
        ("Schema", checks.get("schema", {}).get("passed", False)),
        ("Completeness", checks.get("completeness", {}).get("passed", False)),
        ("Duplicates", checks.get("duplicates", {}).get("passed", False)),
        ("Freshness", checks.get("freshness", {}).get("passed", False)),
    ]
    
    for i, (name, passed) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.metric(name, "✅" if passed else "❌")

    # Detailed Health Table
    st.markdown("### Health Check Details")
    health_details = []
    for check_name, res in checks.items():
        health_details.append({
            "Check": check_name,
            "Passed": "Yes" if res.get("passed") else "No",
            "Details": str(res)
        })
    st.table(pd.DataFrame(health_details))

with tab3:
    st.subheader("Active Alerts")
    alerts = report.get("alerts", [])
    if not alerts:
        st.success("No active alerts. System is stable.")
    else:
        for a in alerts:
            severity = a.get("severity", "info").upper()
            color = "red" if severity == "CRITICAL" else "orange" if severity == "WARNING" else "blue"
            st.markdown(f"**:{color}[{severity}]** {a.get('source')}: {a.get('message')}")
            with st.expander("View Details"):
                st.json(a.get("details", {}))

# Footer
st.divider()
st.caption("Hospital Capacity Forecasting MLOps Dashboard | Built with Streamlit & ZenML")
