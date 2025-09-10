"""Trends visualization page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core.metrics import compute_delta_counts
from core.io_yolo import CLS_MAP


def render_trends_page():
    """Render the Trends page with recognition rate charts."""
    st.title("📈 Recognition Rate Trends")
    
    if 'runs' not in st.session_state or len(st.session_state.runs) < 2:
        st.warning("⚠️ Need at least 2 runs to show trends. Please add more runs.")
        st.stop()
    
    runs = st.session_state.runs
    
    tab1, tab2, tab3 = st.tabs(["📊 Overall Trends", "🔤 Per-Class Accuracy", "📈 Training Δ Counts"])
    
    with tab1:
        render_overall_trends(runs)
    
    with tab2:
        render_per_class_trends(runs)
    
    with tab3:
        render_delta_counts(runs)


def render_overall_trends(runs):
    """Render overall recognition rate trends."""
    st.subheader("Overall Recognition Rate Trends")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_emr = st.checkbox("Show EMR", value=True)
        show_char_acc = st.checkbox("Show Char Accuracy", value=True)
        
        st.divider()
        
        selected_run = st.selectbox(
            "Select run for details",
            options=[run['name'] for run in runs],
            index=len(runs)-1
        )
    
    with col1:
        fig = go.Figure()
        
        x_labels = [run['name'] for run in runs]
        x_times = [run['timestamp'].strftime('%Y-%m-%d %H:%M') for run in runs]
        
        if show_emr:
            emr_values = [run['metrics']['emr'] * 100 for run in runs]
            
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=emr_values,
                mode='lines+markers',
                name='EMR',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>' +
                             'EMR: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        if show_char_acc:
            char_acc_values = [run['metrics']['char_accuracy'] * 100 for run in runs]
            
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=char_acc_values,
                mode='lines+markers',
                name='Character Accuracy',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>' +
                             'Char Acc: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        for i, run in enumerate(runs):
            fig.add_annotation(
                x=x_labels[i],
                y=-5,
                text=x_times[i],
                showarrow=False,
                font=dict(size=9, color='gray'),
                xanchor='center',
                yanchor='top'
            )
        
        fig.update_layout(
            title="Recognition Rate Over Training Runs",
            xaxis_title="Run Name",
            yaxis_title="Recognition Rate (%)",
            yaxis=dict(range=[0, 105]),
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if selected_run:
        run = next(r for r in runs if r['name'] == selected_run)
        
        st.divider()
        st.subheader(f"📋 Run Details: {selected_run}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("EMR", f"{run['metrics']['emr']:.2%}")
        
        with col2:
            st.metric("Char Accuracy", f"{run['metrics']['char_accuracy']:.2%}")
        
        with col3:
            st.metric("Total Images", f"{run['metrics']['n_images']:,}")
        
        with col4:
            st.metric("Edit Distance", f"{run['metrics']['total_edit_distance']:,}")
        
        if run['description']:
            st.info(f"📝 **Description:** {run['description']}")


def render_per_class_trends(runs):
    """Render per-class accuracy trends."""
    st.subheader("Per-Class Character Accuracy Trends")
    
    all_chars = list(CLS_MAP.values())
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**Select Characters**")
        selected_chars = st.multiselect(
            "Choose up to 5 characters",
            options=all_chars,
            default=['A', 'B', '1', '2', '3'],
            max_selections=5,
            help="Select characters to display in the line chart"
        )
    
    with col1:
        if selected_chars:
            fig = go.Figure()
            
            x_labels = [run['name'] for run in runs]
            
            for char in selected_chars:
                y_values = []
                for run in runs:
                    per_class = run['metrics']['per_class_accuracy']
                    if char in per_class:
                        y_values.append(per_class[char]['accuracy'] * 100)
                    else:
                        y_values.append(0)
                
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=y_values,
                    mode='lines+markers',
                    name=f'Char: {char}',
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Per-Class Accuracy Trends",
                xaxis_title="Run Name",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 105]),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("Per-Class Accuracy Heatmap")
    
    heatmap_data = []
    for char in all_chars:
        row = []
        for run in runs:
            per_class = run['metrics']['per_class_accuracy']
            if char in per_class:
                row.append(per_class[char]['accuracy'] * 100)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[run['name'] for run in runs],
        y=all_chars,
        colorscale='RdYlGn',
        zmid=50,
        text=[[f'{val:.1f}%' for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Accuracy (%)")
    ))
    
    fig_heatmap.update_layout(
        title="Character Accuracy Heatmap (Runs × Classes)",
        xaxis_title="Run Name",
        yaxis_title="Character",
        height=800
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)


def render_delta_counts(runs):
    """Render training label delta counts."""
    st.subheader("Training Label Increments (Δ Counts)")
    
    st.markdown("""
    Shows the change in training label counts compared to the previous run.
    - **Positive values** (green): More training samples added
    - **Negative values** (red): Training samples removed
    - **Zero** (white): No change
    """)
    
    delta_data = []
    delta_runs = []
    
    for i, run in enumerate(runs):
        prev_counts = runs[i-1]['train_counts'] if i > 0 else None
        delta = compute_delta_counts(run['train_counts'], prev_counts)
        delta_data.append(delta)
        delta_runs.append(run['name'])
    
    show_positive_only = st.checkbox("Show positive values only", value=False)
    
    heatmap_values = []
    for cls_id in range(36):
        row = []
        for delta in delta_data:
            val = delta.get(cls_id, 0)
            if show_positive_only and val <= 0:
                val = 0
            row.append(val)
        heatmap_values.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_values,
        x=delta_runs,
        y=[CLS_MAP[i] for i in range(36)],
        colorscale='RdBu_r',
        zmid=0,
        text=[[str(int(val)) if val != 0 else '' for val in row] for row in heatmap_values],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Δ Count")
    ))
    
    fig_heatmap.update_layout(
        title="Training Label Δ Counts Heatmap (Runs × Classes)",
        xaxis_title="Run Name",
        yaxis_title="Character",
        height=800
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    st.subheader("Single Run Δ Counts")
    
    selected_run_idx = st.selectbox(
        "Select run for bar chart",
        options=list(range(len(runs))),
        format_func=lambda x: runs[x]['name'],
        index=len(runs)-1 if runs else 0
    )
    
    if selected_run_idx is not None:
        delta = delta_data[selected_run_idx]
        
        df_delta = pd.DataFrame([
            {'Character': CLS_MAP[cls_id], 'Delta': count}
            for cls_id, count in delta.items()
            if count != 0
        ])
        
        if not df_delta.empty:
            df_delta = df_delta.sort_values('Delta', ascending=False)
            
            fig_bar = px.bar(
                df_delta,
                x='Character',
                y='Delta',
                color='Delta',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                title=f"Δ Counts for {runs[selected_run_idx]['name']}",
                labels={'Delta': 'Count Change'},
                height=400
            )
            
            fig_bar.update_layout(
                xaxis_title="Character",
                yaxis_title="Δ Count",
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Added", f"+{sum(v for v in delta.values() if v > 0):,}")
            
            with col2:
                st.metric("Total Removed", f"{sum(v for v in delta.values() if v < 0):,}")
        else:
            st.info("No changes in this run compared to baseline.")