"""Trends visualization page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.metrics import compute_delta_counts
from core.io_yolo import CLS_MAP


def render_trends_page():
    """Render the Trends page with recognition rate charts."""
    st.title("📈 Recognition Rate Trends")
    
    if 'runs' not in st.session_state or len(st.session_state.runs) == 0:
        st.info("📊 No runs available yet. Please add runs to view trends.")
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
        show_plate_accuracy = st.checkbox("Show Plate Accuracy", value=True)
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
        
        # Calculate deltas for plates and characters
        plates_deltas = []
        chars_deltas = []
        
        for i, run in enumerate(runs):
            n_plates = run['metrics'].get('n_plates', 0)
            total_chars = run['metrics'].get('total_gt_chars', 0)
            
            if i == 0:
                plates_deltas.append((n_plates, None))
                chars_deltas.append((total_chars, None))
            else:
                prev_plates = runs[i-1]['metrics'].get('n_plates', 0)
                prev_chars = runs[i-1]['metrics'].get('total_gt_chars', 0)
                plates_deltas.append((n_plates, n_plates - prev_plates))
                chars_deltas.append((total_chars, total_chars - prev_chars))
        
        if show_plate_accuracy:
            plate_accuracy_values = [run['metrics']['plate_accuracy'] * 100 for run in runs]
            
            # Build custom hover text with plate info
            hover_texts = []
            for i, plate_acc in enumerate(plate_accuracy_values):
                plates_total, plates_delta = plates_deltas[i]
                
                hover_text = f'Plate Accuracy: {plate_acc:.2f}%<br>'
                hover_text += f'車牌數: {plates_total:,}'
                if plates_delta is not None:
                    hover_text += f' ({plates_delta:+,})'
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=plate_accuracy_values,
                mode='lines+markers',
                name='Plate Accuracy',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts
            ))
        
        if show_char_acc:
            char_acc_values = [run['metrics']['char_accuracy'] * 100 for run in runs]
            
            # Build custom hover text with character info
            hover_texts = []
            for i, char_acc in enumerate(char_acc_values):
                chars_total, chars_delta = chars_deltas[i]
                
                hover_text = f'Char Acc: {char_acc:.2f}%<br>'
                hover_text += f'字元數: {chars_total:,}'
                if chars_delta is not None:
                    hover_text += f' ({chars_delta:+,})'
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=char_acc_values,
                mode='lines+markers',
                name='Character Accuracy',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=8),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts
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
            st.metric("Plate Accuracy", f"{run['metrics']['plate_accuracy']:.2%}")
        
        with col2:
            st.metric("Char Accuracy", f"{run['metrics']['char_accuracy']:.2%}")
        
        with col3:
            st.metric("Total Plates", f"{run['metrics']['n_plates']:,}")
        
        with col4:
            st.metric("Edit Distance", f"{run['metrics']['total_edit_distance']:,}")
        
        if run['description']:
            st.info(f"📝 **Description:** {run['description']}")


def render_per_class_trends(runs):
    """Render per-class accuracy trends."""
    st.subheader("Per-Class Character Accuracy Analysis")
    
    if len(runs) == 0:
        st.info("No runs available for analysis")
        return
    
    # Select run for detailed view
    selected_run_idx = st.selectbox(
        "Select run for detailed analysis",
        options=list(range(len(runs))),
        format_func=lambda x: runs[x]['name'],
        index=len(runs)-1 if runs else 0,
        help="Choose a training run to view its per-class accuracy and training data distribution"
    )
    
    if selected_run_idx is None:
        return
    
    selected_run = runs[selected_run_idx]
    prev_run = runs[selected_run_idx - 1] if selected_run_idx > 0 else None
    
    # Prepare data for 37 classes (plates + 36 characters)
    class_labels = ['Plates'] + [CLS_MAP[i] for i in range(36)]
    
    # Calculate previous counts (for stacked bar chart)
    prev_counts = prev_run['train_counts'] if prev_run else {}
    curr_counts = selected_run['train_counts']
    
    # Prepare data arrays
    base_counts = []  # Previous training counts
    delta_counts = []  # New additions
    accuracies = []   # Accuracy values
    prev_accuracies = []  # Previous accuracy values for comparison
    accuracy_colors = []  # Colors for K-line style
    hover_texts = []  # Custom hover text
    
    # Handle plates data
    n_plates = selected_run['metrics'].get('n_plates', 0)
    prev_plates = prev_run['metrics'].get('n_plates', 0) if prev_run else 0
    base_counts.append(prev_plates)
    delta_counts.append(n_plates - prev_plates)
    
    curr_plate_acc = selected_run['metrics']['plate_accuracy'] * 100
    prev_plate_acc = prev_run['metrics']['plate_accuracy'] * 100 if prev_run else curr_plate_acc
    accuracies.append(curr_plate_acc)
    prev_accuracies.append(prev_plate_acc)
    
    # Determine color and hover text based on change
    if not prev_run:
        # First run - always gray
        accuracy_colors.append('#888888')
        hover_text = f"Plates<br>Baseline: {curr_plate_acc:.1f}%<br>(Initial run)"
    elif curr_plate_acc > prev_plate_acc:
        accuracy_colors.append('#FF4444')  # Red for increase
        change = f"+{curr_plate_acc - prev_plate_acc:.1f}%"
        hover_text = f"Plates<br>Current: {curr_plate_acc:.1f}%<br>Previous: {prev_plate_acc:.1f}%<br>Change: {change}"
    elif curr_plate_acc < prev_plate_acc:
        accuracy_colors.append('#44AA44')  # Green for decrease
        change = f"{curr_plate_acc - prev_plate_acc:.1f}%"
        hover_text = f"Plates<br>Current: {curr_plate_acc:.1f}%<br>Previous: {prev_plate_acc:.1f}%<br>Change: {change}"
    else:
        accuracy_colors.append('#888888')  # Gray for no change
        hover_text = f"Plates<br>Current: {curr_plate_acc:.1f}%<br>Previous: {prev_plate_acc:.1f}%<br>No change"
    
    hover_texts.append(hover_text)
    
    # Handle character data
    per_class_acc = selected_run['metrics']['per_class_accuracy']
    prev_per_class_acc = prev_run['metrics']['per_class_accuracy'] if prev_run else {}
    
    for i in range(36):
        char = CLS_MAP[i]
        
        # Get counts
        prev_count = prev_counts.get(i, prev_counts.get(str(i), 0)) if prev_counts else 0
        curr_count = curr_counts.get(i, curr_counts.get(str(i), 0))
        
        base_counts.append(prev_count)
        delta_counts.append(curr_count - prev_count)
        
        # Get accuracy
        curr_acc = per_class_acc[char]['accuracy'] * 100 if char in per_class_acc else 0
        
        # Handle previous accuracy based on whether prev_run exists
        if prev_run:
            # If there's a previous run, try to get the previous accuracy
            prev_acc = prev_per_class_acc[char]['accuracy'] * 100 if char in prev_per_class_acc else 0
        else:
            # For the first run, start point equals end point
            prev_acc = curr_acc
        
        accuracies.append(curr_acc)
        prev_accuracies.append(prev_acc)
        
        # Determine color based on change
        if not prev_run:
            # First run - always gray
            accuracy_colors.append('#888888')
            hover_text = f"{char}<br>Baseline: {curr_acc:.1f}%<br>(Initial run)"
        elif curr_acc > prev_acc:
            accuracy_colors.append('#FF4444')  # Red for increase
            change = f"+{curr_acc - prev_acc:.1f}%"
            hover_text = f"{char}<br>Current: {curr_acc:.1f}%<br>Previous: {prev_acc:.1f}%<br>Change: {change}"
        elif curr_acc < prev_acc:
            accuracy_colors.append('#44AA44')  # Green for decrease
            change = f"{curr_acc - prev_acc:.1f}%"
            hover_text = f"{char}<br>Current: {curr_acc:.1f}%<br>Previous: {prev_acc:.1f}%<br>Change: {change}"
        else:
            accuracy_colors.append('#888888')  # Gray for no change
            hover_text = f"{char}<br>Current: {curr_acc:.1f}%<br>Previous: {prev_acc:.1f}%<br>No change"
        
        hover_texts.append(hover_text)
    
    # Round all accuracy values to avoid floating point issues
    # This ensures proper color display in candlestick chart
    accuracies = [round(acc, 1) for acc in accuracies]
    prev_accuracies = [round(acc, 1) for acc in prev_accuracies]
    
    # Ensure truly equal values for neutral cases
    # After rounding, values within 0.1% are considered equal
    adjusted_prev_accuracies = []
    for i in range(len(prev_accuracies)):
        if abs(prev_accuracies[i] - accuracies[i]) < 0.1:  # Less than 0.1% difference
            adjusted_prev_accuracies.append(accuracies[i])  # Make them exactly equal
        else:
            adjusted_prev_accuracies.append(prev_accuracies[i])
    
    # Create dual-axis plot
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add stacked bar chart for counts
    # Base layer (previous counts)
    fig.add_trace(
        go.Bar(
            x=class_labels,
            y=base_counts,
            name='Previous Training Count',
            marker_color='darkblue',
            hovertemplate='Previous: %{y:,.0f}<extra></extra>',
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Delta layer (new additions)
    fig.add_trace(
        go.Bar(
            x=class_labels,
            y=delta_counts,
            name='New Additions',
            marker_color='lightblue',
            hovertemplate='Added: %{y:,.0f}<extra></extra>',
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Add Candlestick chart for accuracy (K-line style)
    # Simple and clean implementation
    fig.add_trace(
        go.Candlestick(
            x=class_labels,
            open=adjusted_prev_accuracies,
            close=accuracies,
            high=[max(p, c) for p, c in zip(adjusted_prev_accuracies, accuracies)],
            low=[min(p, c) for p, c in zip(adjusted_prev_accuracies, accuracies)],
            increasing_line_color='#FF4444',  # Red for increase
            decreasing_line_color='#44AA44',  # Green for decrease
            name='Accuracy (K-line)',
            text=hover_texts,
            hoverinfo='text',
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"Per-Class Analysis for {selected_run['name']}",
        barmode='stack',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Class", tickangle=-45)
    fig.update_yaxes(title_text="Training Sample Count", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 105])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_samples = sum(base_counts) + sum(delta_counts)
        st.metric("Total Training Samples", f"{total_samples:,}")
    
    with col2:
        total_new = sum(delta_counts)
        st.metric("New Additions", f"{total_new:,}")
    
    with col3:
        # Count accuracy changes
        improved = sum(1 for curr, prev in zip(accuracies, prev_accuracies) if curr > prev)
        st.metric("Classes Improved", improved, delta=f"↑ {improved}", delta_color="normal")
    
    with col4:
        declined = sum(1 for curr, prev in zip(accuracies, prev_accuracies) if curr < prev)
        st.metric("Classes Declined", declined, delta=f"↓ {declined}", delta_color="inverse")
    
    with col5:
        perfect_classes = sum(1 for acc in accuracies if acc >= 95)
        st.metric("Classes ≥95%", perfect_classes)
    
    st.divider()
    st.subheader("Character Accuracy Heatmap (All Runs)")
    
    all_chars = list(CLS_MAP.values())
    
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
            # Handle both string and integer keys
            val = delta.get(cls_id, delta.get(str(cls_id), 0))
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
            {'Character': CLS_MAP[int(cls_id) if isinstance(cls_id, str) else cls_id], 'Delta': count}
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