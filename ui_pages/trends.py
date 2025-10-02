"""Trends visualization page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.metrics import compute_delta_counts
from core.io_yolo import CLS_MAP, get_class_counts


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
        use_dynamic_yaxis = st.checkbox(
            "Dynamic Y-axis Scale",
            value=True,
            help="Adjust Y-axis range based on actual data for better contrast"
        )

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
        plate_accuracy_values = []
        char_acc_values = []

        for i, run in enumerate(runs):
            # Use training set counts for consistency
            n_plates = run.get('n_train_plates', 0)
            total_chars = run.get('n_train_chars', sum(run['train_counts'].values()))

            if i == 0:
                plates_deltas.append((n_plates, None))
                chars_deltas.append((total_chars, None))
            else:
                prev_plates = runs[i-1].get('n_train_plates', 0)
                prev_chars = runs[i-1].get('n_train_chars', sum(runs[i-1]['train_counts'].values()))
                plates_deltas.append((n_plates, n_plates - prev_plates))
                chars_deltas.append((total_chars, total_chars - prev_chars))
        
        if show_char_acc:
            char_acc_values = [run['metrics']['char_accuracy'] * 100 for run in runs]

            # Build custom hover text with character info
            hover_texts = []
            for i, char_acc in enumerate(char_acc_values):
                chars_total, chars_delta = chars_deltas[i]

                # Calculate accuracy delta
                if i > 0:
                    prev_char_acc = char_acc_values[i - 1]
                    acc_delta = char_acc - prev_char_acc
                    hover_text = f'Char Acc: {char_acc:.2f}% ({acc_delta:+.2f}%)<br>'
                else:
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

        if show_plate_accuracy:
            plate_accuracy_values = [run['metrics']['plate_accuracy'] * 100 for run in runs]

            # Build custom hover text with plate info
            hover_texts = []
            for i, plate_acc in enumerate(plate_accuracy_values):
                plates_total, plates_delta = plates_deltas[i]

                # Calculate accuracy delta
                if i > 0:
                    prev_plate_acc = plate_accuracy_values[i - 1]
                    acc_delta = plate_acc - prev_plate_acc
                    hover_text = f'Plate Accuracy: {plate_acc:.2f}% ({acc_delta:+.2f}%)<br>'
                else:
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

        # Calculate Y-axis range
        if use_dynamic_yaxis:
            # Collect all displayed values
            all_values = []
            if show_plate_accuracy:
                all_values.extend(plate_accuracy_values)
            if show_char_acc:
                all_values.extend(char_acc_values)

            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                # Add padding (5% of range, minimum 2%)
                y_range = y_max - y_min
                padding = max(y_range * 0.05, 2)
                y_axis_range = [max(0, y_min - padding), min(100, y_max + padding)]
            else:
                # Fallback to fixed range if no data
                y_axis_range = [0, 105]
        else:
            # Fixed range
            y_axis_range = [0, 105]

        fig.update_layout(
            title="Recognition Rate Over Training Runs",
            xaxis_title="Run Name",
            yaxis_title="Recognition Rate (%)",
            yaxis=dict(range=y_axis_range),
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
            st.metric("Total Plates", f"{run.get('n_train_plates', 0):,}")
        
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
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_run_idx = st.selectbox(
            "Select run for detailed analysis",
            options=list(range(len(runs))),
            format_func=lambda x: runs[x]['name'],
            index=len(runs)-1 if runs else 0,
            help="Choose a training run to view its per-class accuracy and training data distribution"
        )

    with col2:
        sort_option = st.selectbox(
            "Sort by",
            options=["Default Order", "Accuracy", "Data Count", "GT Data Count"],
            index=0,
            help="Sort character classes by accuracy, training data count, or GT data count. Plates always remain first."
        )

        show_training_in_slider = st.checkbox(
            "Show Training Data in Range Slider",
            value=False,
            help="Show training data (Previous + New) in the bottom navigation slider. When unchecked, only GT distribution is shown."
        )

        # Check if GT data is available
        has_gt_data = 'gt_labels' in st.session_state and st.session_state.gt_labels
        show_gt_distribution = has_gt_data

        if not has_gt_data:
            st.info("💡 Upload GT data to compare distributions")

    # Add class selection with checkboxes
    st.divider()
    all_classes = ['Plates'] + [CLS_MAP[i] for i in range(36)]

    # Initialize session state for checkboxes if not present
    for cls in all_classes:
        key = f"class_check_{cls}"
        if key not in st.session_state:
            # Default to all selected except I and O
            st.session_state[key] = False if cls in ['I', 'O'] else True

    with st.expander("📊 Select classes to display", expanded=False):
        # Control buttons and selection count
        col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("Select All", use_container_width=True, key="select_all_btn"):
                for cls in all_classes:
                    st.session_state[f"class_check_{cls}"] = True
                st.rerun()
        with col_btn2:
            if st.button("Clear All", use_container_width=True, key="clear_all_btn"):
                for cls in all_classes:
                    st.session_state[f"class_check_{cls}"] = False
                st.rerun()
        with col_info:
            # Count selected classes
            selected_count = sum(1 for cls in all_classes if st.session_state.get(f"class_check_{cls}", False if cls in ['I', 'O'] else True))
            st.info(f"📌 Selected: {selected_count}/{len(all_classes)} classes")

        st.markdown("---")  # Divider line

        # Row 1: Plates only
        st.checkbox("🚗 **Plates**", key="class_check_Plates")

        # Row 2: Numbers 0-9
        st.markdown("**Numbers:**")
        num_cols = st.columns(10)
        for i in range(10):
            with num_cols[i]:
                st.checkbox(str(i), key=f"class_check_{i}")

        # Row 3: Letters A-Z
        st.markdown("**Letters:**")
        # First row of letters: A-M (13 letters)
        letter_cols1 = st.columns(13)
        for i in range(13):
            char_idx = i + 10  # A starts at index 10
            with letter_cols1[i]:
                st.checkbox(CLS_MAP[char_idx], key=f"class_check_{CLS_MAP[char_idx]}")

        # Second row of letters: N-Z (13 letters)
        letter_cols2 = st.columns(13)
        for i in range(13):
            char_idx = i + 23  # N starts at index 23
            if char_idx < 36:  # Make sure we don't exceed Z
                with letter_cols2[i]:
                    st.checkbox(CLS_MAP[char_idx], key=f"class_check_{CLS_MAP[char_idx]}")

    # Collect selected classes
    selected_classes = []
    if st.session_state.get("class_check_Plates", True):
        selected_classes.append("Plates")
    for i in range(36):
        cls_name = CLS_MAP[i]
        default_selected = False if cls_name in ['I', 'O'] else True
        if st.session_state.get(f"class_check_{cls_name}", default_selected):
            selected_classes.append(cls_name)

    if selected_run_idx is None or not selected_classes:
        if not selected_classes:
            st.info("Please select at least one class to display.")
        return
    
    selected_run = runs[selected_run_idx]
    prev_run = runs[selected_run_idx - 1] if selected_run_idx > 0 else None

    # Calculate previous counts (for stacked bar chart)
    prev_counts = prev_run['train_counts'] if prev_run else {}
    curr_counts = selected_run['train_counts']

    # Calculate GT distribution if available and requested
    gt_counts = {}
    gt_n_plates = 0
    if show_gt_distribution and has_gt_data:
        gt_labels = st.session_state.gt_labels
        gt_counts = get_class_counts(gt_labels)
        gt_n_plates = len(gt_labels)

    # Get per-class accuracy for sorting
    per_class_acc = selected_run['metrics']['per_class_accuracy']

    # Filter character indices based on selected classes
    all_char_indices = list(range(36))
    filtered_char_indices = [i for i in all_char_indices if CLS_MAP[i] in selected_classes]

    # Sort filtered character indices based on selected option
    if sort_option == "Accuracy":
        # Sort by accuracy (descending)
        filtered_char_indices.sort(key=lambda i: per_class_acc[CLS_MAP[i]]['accuracy'] * 100 if CLS_MAP[i] in per_class_acc else 0, reverse=True)
    elif sort_option == "Data Count":
        # Sort by training data count (descending)
        filtered_char_indices.sort(key=lambda i: curr_counts.get(i, curr_counts.get(str(i), 0)), reverse=True)
    elif sort_option == "GT Data Count":
        # Sort by GT data count (descending)
        filtered_char_indices.sort(key=lambda i: gt_counts.get(i, gt_counts.get(str(i), 0)), reverse=True)
    # else: keep default order

    # Prepare class labels based on selection
    # Plates always first if selected, then sorted characters
    class_labels = []
    if 'Plates' in selected_classes:
        class_labels.append('Plates')
    class_labels.extend([CLS_MAP[i] for i in filtered_char_indices])

    # Prepare data arrays
    base_counts = []  # Previous training counts
    delta_counts = []  # New additions
    gt_distribution = []  # GT test set counts
    accuracies = []   # Accuracy values
    prev_accuracies = []  # Previous accuracy values for comparison
    accuracy_colors = []  # Colors for K-line style
    hover_texts = []  # Custom hover text

    # Handle plates data if selected
    if 'Plates' in selected_classes:
        n_plates = selected_run.get('n_train_plates', 0)
        prev_plates = prev_run.get('n_train_plates', 0) if prev_run else 0
        base_counts.append(prev_plates)
        delta_counts.append(n_plates - prev_plates)
        gt_distribution.append(gt_n_plates if show_gt_distribution else 0)

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
    prev_per_class_acc = prev_run['metrics']['per_class_accuracy'] if prev_run else {}

    # Process only filtered characters in sorted order
    for i in filtered_char_indices:
        char = CLS_MAP[i]

        # Get counts
        prev_count = prev_counts.get(i, prev_counts.get(str(i), 0)) if prev_counts else 0
        curr_count = curr_counts.get(i, curr_counts.get(str(i), 0))

        base_counts.append(prev_count)
        delta_counts.append(curr_count - prev_count)

        # Get GT count
        gt_count = gt_counts.get(i, gt_counts.get(str(i), 0)) if show_gt_distribution else 0
        gt_distribution.append(gt_count)
        
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
    
    # Update layout with selected classes count
    title_suffix = f" ({len(selected_classes)} selected)" if len(selected_classes) < 37 else ""
    fig.update_layout(
        title=f"Per-Class Analysis for {selected_run['name']}{title_suffix}",
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
    fig.update_xaxes(
        title_text="Class",
        tickangle=-45,
        rangeslider_visible=False  # Disable built-in rangeslider
    )
    fig.update_yaxes(title_text="Training Sample Count", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy (%)", secondary_y=True, range=[0, 105])

    st.plotly_chart(fig, use_container_width=True)

    # Add custom overview bar chart below main chart
    if show_gt_distribution:
        fig_overview = go.Figure()

        # Add training data if checkbox is enabled
        if show_training_in_slider:
            # Base layer (previous counts)
            fig_overview.add_trace(go.Bar(
                x=class_labels,
                y=base_counts,
                name='Previous Training Count',
                marker_color='darkblue',
                hovertemplate='Previous: %{y:,.0f}<extra></extra>'
            ))

            # Delta layer (new additions)
            fig_overview.add_trace(go.Bar(
                x=class_labels,
                y=delta_counts,
                name='New Additions',
                marker_color='lightblue',
                hovertemplate='Added: %{y:,.0f}<extra></extra>'
            ))

        # GT distribution (always shown)
        fig_overview.add_trace(go.Bar(
            x=class_labels,
            y=gt_distribution,
            name='GT Distribution',
            marker_color='orange',
            marker_pattern_shape='/' if show_training_in_slider else '',  # Pattern only when training data shown
            opacity=0.8,
            hovertemplate='GT: %{y:,.0f}<extra></extra>'
        ))

        # Update layout for overview chart
        overview_title = "Data Distribution Overview"
        if not show_training_in_slider:
            overview_title += " (GT Only)"

        fig_overview.update_layout(
            title=overview_title,
            barmode='stack',
            height=200,  # Smaller height for overview
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=40, b=60)
        )

        fig_overview.update_xaxes(title_text="", tickangle=-45, tickfont=dict(size=8))
        y_overview_title = "GT Count" if not show_training_in_slider else "Sample Count"
        fig_overview.update_yaxes(title_text=y_overview_title)

        st.plotly_chart(fig_overview, use_container_width=True)
    
    # Display summary metrics
    # Separate plates and chars statistics
    has_plates = 'Plates' in selected_classes

    if has_plates:
        # Plates statistics (first element)
        plates_base = base_counts[0]
        plates_delta = delta_counts[0]
        plates_gt = gt_distribution[0] if show_gt_distribution else 0
        plates_total = plates_base + plates_delta

        # Chars statistics (remaining elements)
        chars_base = base_counts[1:]
        chars_delta = delta_counts[1:]
        chars_gt = gt_distribution[1:] if show_gt_distribution else []
        chars_total = sum(chars_base) + sum(chars_delta)
        chars_new = sum(chars_delta)
        chars_gt_total = sum(chars_gt) if show_gt_distribution else 0

        # Accuracy statistics for chars only
        chars_accuracies = accuracies[1:]
        chars_prev_accuracies = prev_accuracies[1:]
    else:
        # No plates, all data is chars
        chars_base = base_counts
        chars_delta = delta_counts
        chars_gt = gt_distribution if show_gt_distribution else []
        chars_total = sum(chars_base) + sum(chars_delta)
        chars_new = sum(chars_delta)
        chars_gt_total = sum(chars_gt) if show_gt_distribution else 0

        # Accuracy statistics for all selected classes
        chars_accuracies = accuracies
        chars_prev_accuracies = prev_accuracies

    # Row 1: Plates statistics (if selected)
    if has_plates:
        st.markdown("**🚗 Plates Statistics**")
        if show_gt_distribution:
            plate_cols = st.columns(3)
        else:
            plate_cols = st.columns(2)

        with plate_cols[0]:
            st.metric("Total Training Plates", f"{plates_total:,}",
                      help="Total training plate count")

        with plate_cols[1]:
            st.metric("New Plate Additions", f"{plates_delta:,}",
                      help="New plates added in this run")

        if show_gt_distribution:
            with plate_cols[2]:
                st.metric("Total GT Plates", f"{plates_gt:,}",
                          help="Total GT test plates")

        st.divider()

    # Row 2: Chars statistics
    st.markdown("**🔤 Character Statistics**")
    if show_gt_distribution:
        char_cols = st.columns(6)
    else:
        char_cols = st.columns(5)

    with char_cols[0]:
        st.metric("Total Training Chars", f"{chars_total:,}",
                  help=f"Total character samples for selected classes")

    with char_cols[1]:
        st.metric("New Char Additions", f"{chars_new:,}",
                  help="New character samples added in this run")

    with char_cols[2]:
        # Count accuracy changes for chars
        improved = sum(1 for curr, prev in zip(chars_accuracies, chars_prev_accuracies) if curr > prev)
        st.metric("Classes Improved", improved, delta=improved, delta_color="inverse")

    with char_cols[3]:
        declined = sum(1 for curr, prev in zip(chars_accuracies, chars_prev_accuracies) if curr < prev)
        st.metric("Classes Declined", declined, delta=declined, delta_color="normal")

    with char_cols[4]:
        perfect_classes = sum(1 for acc in chars_accuracies if acc >= 95)
        prev_perfect_classes = sum(1 for acc in chars_prev_accuracies if acc >= 95) if prev_run else perfect_classes
        delta_perfect = perfect_classes - prev_perfect_classes
        st.metric("Classes ≥95%", f"{perfect_classes}/{len(chars_accuracies)}",
                  delta=delta_perfect, delta_color="inverse")

    if show_gt_distribution:
        with char_cols[5]:
            st.metric("Total GT Chars", f"{chars_gt_total:,}",
                      help="Total GT test character samples")
    
    st.divider()
    st.subheader("Character Accuracy Heatmap (All Runs)")

    # Color scale option
    use_dynamic_scale = st.checkbox(
        "Dynamic Color Scale",
        value=False,
        help="Adjust color scale based on actual data range for better contrast"
    )

    # Use filtered and sorted classes (same as Per-Class Analysis Chart)
    heatmap_classes = []
    heatmap_data = []

    # Add Plates row first if selected
    if 'Plates' in selected_classes:
        heatmap_classes.append('Plates')
        plates_row = []
        for run in runs:
            plate_acc = run['metrics']['plate_accuracy'] * 100
            plates_row.append(plate_acc)
        heatmap_data.append(plates_row)

    # Add character rows for selected classes only (using filtered indices)
    for i in filtered_char_indices:
        char = CLS_MAP[i]
        heatmap_classes.append(char)
        row = []
        for run in runs:
            per_class = run['metrics']['per_class_accuracy']
            if char in per_class:
                row.append(per_class[char]['accuracy'] * 100)
            else:
                row.append(0)
        heatmap_data.append(row)

    # Calculate color scale parameters
    if use_dynamic_scale and heatmap_data:
        # Flatten all data to find min/max
        all_values = [val for row in heatmap_data for val in row]
        zmin = min(all_values)
        zmax = max(all_values)
        zmid = (zmin + zmax) / 2
    else:
        # Fixed scale: 0-100%
        zmin = 0
        zmax = 100
        zmid = 50

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[run['name'] for run in runs],
        y=heatmap_classes,
        colorscale='RdYlGn',
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        text=[[f'{val:.1f}%' for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Accuracy (%)")
    ))

    # Update title with selected classes count
    heatmap_title_suffix = f" ({len(selected_classes)} selected)" if len(selected_classes) < 37 else ""
    fig_heatmap.update_layout(
        title=f"Character & Plate Accuracy Heatmap (Runs × Classes){heatmap_title_suffix}",
        xaxis_title="Run Name",
        yaxis_title="Class",
        height=850
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)


def render_delta_counts(runs):
    """Render training label delta counts."""
    st.subheader("Training Label Increments (Δ Counts)")
    
    st.markdown("""
    Shows the change in training label counts compared to the previous run.

    **Note**: First run is not shown as there's no previous data to compare.
    """)
    
    delta_data = []
    plate_deltas = []
    delta_runs = []

    # Skip first run since there's no previous data to compare
    for i in range(1, len(runs)):
        run = runs[i]
        prev_counts = runs[i-1]['train_counts']
        delta = compute_delta_counts(run['train_counts'], prev_counts)

        # Calculate plate count delta (using training plates)
        n_plates = run.get('n_train_plates', 0)
        prev_plates = runs[i-1].get('n_train_plates', 0)
        plate_delta = n_plates - prev_plates

        delta_data.append(delta)
        plate_deltas.append(plate_delta)
        delta_runs.append(run['name'])
    
    # Add sorting option
    st.divider()
    sort_option = st.selectbox(
        "Sort by",
        options=[
            "Default Order (0-9, A-Z)",
            "Latest Delta (most recent run)",
            "Total Delta (sum across all runs)",
            "Absolute Change (largest magnitude)"
        ],
        index=0,
        help="Note: Plates always remain first if selected.",
        key="delta_sort_option"
    )

    # Add class selection UI
    all_classes = ['Plates'] + [CLS_MAP[i] for i in range(36)]

    # Initialize session state for delta counts checkboxes if not present
    for cls in all_classes:
        key = f"delta_class_check_{cls}"
        if key not in st.session_state:
            # Default to all selected except I and O
            st.session_state[key] = False if cls in ['I', 'O'] else True

    with st.expander("📊 Select classes to display", expanded=False):
        # Control buttons and selection count
        col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("Select All", use_container_width=True, key="delta_select_all_btn"):
                for cls in all_classes:
                    st.session_state[f"delta_class_check_{cls}"] = True
                st.rerun()
        with col_btn2:
            if st.button("Clear All", use_container_width=True, key="delta_clear_all_btn"):
                for cls in all_classes:
                    st.session_state[f"delta_class_check_{cls}"] = False
                st.rerun()
        with col_info:
            # Count selected classes
            selected_count = sum(1 for cls in all_classes if st.session_state.get(f"delta_class_check_{cls}", False if cls in ['I', 'O'] else True))
            st.info(f"📌 Selected: {selected_count}/{len(all_classes)} classes")

        st.markdown("---")  # Divider line

        # Row 1: Plates only
        st.checkbox("🚗 **Plates**", key="delta_class_check_Plates")

        # Row 2: Numbers 0-9
        st.markdown("**Numbers:**")
        num_cols = st.columns(10)
        for i in range(10):
            with num_cols[i]:
                st.checkbox(str(i), key=f"delta_class_check_{i}")

        # Row 3: Letters A-Z
        st.markdown("**Letters:**")
        # First row of letters: A-M (13 letters)
        letter_cols1 = st.columns(13)
        for i in range(13):
            char_idx = i + 10  # A starts at index 10
            with letter_cols1[i]:
                st.checkbox(CLS_MAP[char_idx], key=f"delta_class_check_{CLS_MAP[char_idx]}")

        # Second row of letters: N-Z (13 letters)
        letter_cols2 = st.columns(13)
        for i in range(13):
            char_idx = i + 23  # N starts at index 23
            if char_idx < 36:  # Make sure we don't exceed Z
                with letter_cols2[i]:
                    st.checkbox(CLS_MAP[char_idx], key=f"delta_class_check_{CLS_MAP[char_idx]}")

    # Collect selected classes
    selected_classes = []
    if st.session_state.get("delta_class_check_Plates", True):
        selected_classes.append("Plates")
    selected_char_indices = []
    for i in range(36):
        cls_name = CLS_MAP[i]
        default_selected = False if cls_name in ['I', 'O'] else True
        if st.session_state.get(f"delta_class_check_{cls_name}", default_selected):
            selected_classes.append(cls_name)
            selected_char_indices.append(i)

    # Sort selected character indices based on sort option
    if delta_data and not sort_option.startswith("Default Order"):
        if sort_option.startswith("Latest Delta"):
            # Sort by latest run's delta (descending)
            latest_delta = delta_data[-1]
            selected_char_indices.sort(
                key=lambda i: latest_delta.get(i, latest_delta.get(str(i), 0)),
                reverse=True
            )
        elif sort_option.startswith("Total Delta"):
            # Sort by sum of all deltas (descending)
            selected_char_indices.sort(
                key=lambda i: sum(d.get(i, d.get(str(i), 0)) for d in delta_data),
                reverse=True
            )
        elif sort_option.startswith("Absolute Change"):
            # Sort by absolute value of latest delta (descending)
            latest_delta = delta_data[-1]
            selected_char_indices.sort(
                key=lambda i: abs(latest_delta.get(i, latest_delta.get(str(i), 0))),
                reverse=True
            )

    if not selected_classes:
        st.info("Please select at least one class to display.")
        return

    show_positive_only = st.checkbox("Show positive values only", value=False, key="delta_show_positive_only")
    use_dynamic_scale = st.checkbox(
        "Dynamic Color Scale",
        value=False,
        help="Adjust color scale based on actual data range for better contrast",
        key="delta_dynamic_scale"
    )
    
    # Add plates row first (if selected)
    heatmap_values = []
    y_labels = []

    if 'Plates' in selected_classes:
        plate_row = []
        for plate_delta in plate_deltas:
            val = plate_delta
            if show_positive_only and val <= 0:
                val = 0
            plate_row.append(val)
        heatmap_values.append(plate_row)
        y_labels.append('Plates')

    # Add character rows (only for selected classes)
    for cls_id in selected_char_indices:
        row = []
        for delta in delta_data:
            # Handle both string and integer keys
            val = delta.get(cls_id, delta.get(str(cls_id), 0))
            if show_positive_only and val <= 0:
                val = 0
            row.append(val)
        heatmap_values.append(row)
        y_labels.append(CLS_MAP[cls_id])
    
    # Calculate color scale parameters
    if use_dynamic_scale and heatmap_values:
        # Flatten all data to find min/max
        all_values = [val for row in heatmap_values for val in row]
        if all_values:
            zmin = min(all_values)
            zmax = max(all_values)
        else:
            zmin = None
            zmax = None
    else:
        # Auto scale (Plotly default)
        zmin = None
        zmax = None

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_values,
        x=delta_runs,
        y=y_labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        text=[[str(int(val)) if val != 0 else '' for val in row] for row in heatmap_values],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Δ Count")
    ))

    # Update title with selected classes count
    title_suffix = f" ({len(selected_classes)} selected)" if len(selected_classes) < 37 else ""
    fig_heatmap.update_layout(
        title=f"Training Label Δ Counts Heatmap (Runs × Classes){title_suffix}",
        xaxis_title="Run Name",
        yaxis_title="Character",
        height=800
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    st.subheader("Single Run Δ Counts")
    
    # Adjust for skipping first run (delta_data starts from index 1 of runs)
    if len(delta_data) > 0:
        col_select, col_scale = st.columns([3, 1])

        with col_select:
            selected_run_idx = st.selectbox(
                "Select run for bar chart",
                options=list(range(len(delta_data))),
                format_func=lambda x: runs[x+1]['name'],  # Offset by 1 since we skip first run
                index=len(delta_data)-1 if delta_data else 0
            )

        with col_scale:
            use_dynamic_scale_bar = st.checkbox(
                "Dynamic Color Scale",
                value=False,
                help="Adjust color scale based on actual data range for better contrast",
                key="delta_bar_dynamic_scale"
            )

        if selected_run_idx is not None:
            delta = delta_data[selected_run_idx]
            plate_delta = plate_deltas[selected_run_idx]

            # Create DataFrame with plates first, then characters (following sort order)
            delta_items = []

            # Add plate delta if non-zero and selected
            if plate_delta != 0 and 'Plates' in selected_classes:
                delta_items.append({'Character': 'Plates', 'Delta': plate_delta})

            # Add character deltas in the same order as the sorted indices
            for cls_id in selected_char_indices:
                val = delta.get(cls_id, delta.get(str(cls_id), 0))
                if val != 0:
                    char_name = CLS_MAP[cls_id]
                    delta_items.append({'Character': char_name, 'Delta': val})

        df_delta = pd.DataFrame(delta_items)

        if not df_delta.empty:
            # Calculate color scale range
            if use_dynamic_scale_bar:
                delta_values = df_delta['Delta'].values
                color_min = min(delta_values)
                color_max = max(delta_values)
            else:
                color_min = None
                color_max = None

            fig_bar = px.bar(
                df_delta,
                x='Character',
                y='Delta',
                color='Delta',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                range_color=[color_min, color_max] if use_dynamic_scale_bar else None,
                title=f"Δ Counts for {runs[selected_run_idx+1]['name']}",
                labels={'Delta': 'Count Change'},
                height=400
            )
            
            fig_bar.update_layout(
                xaxis_title="Character",
                yaxis_title="Δ Count",
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Separate plates and chars statistics (only for selected classes)
            if 'Plates' in selected_classes:
                st.markdown("**🚗 Plates Statistics**")
                plate_cols = st.columns(2)

                with plate_cols[0]:
                    plate_added = plate_delta if plate_delta > 0 else 0
                    st.metric("Plates Added", f"+{plate_added:,}")

                with plate_cols[1]:
                    plate_removed = plate_delta if plate_delta < 0 else 0
                    st.metric("Plates Removed", f"{plate_removed:,}")

                st.divider()

            st.markdown("**🔤 Character Statistics**")
            char_cols = st.columns(2)

            with char_cols[0]:
                # Only count selected characters
                char_added = sum(v for cls_id, v in delta.items()
                               if v is not None and v > 0
                               and CLS_MAP[int(cls_id) if isinstance(cls_id, str) else cls_id] in selected_classes)
                st.metric("Chars Added", f"+{char_added:,}")

            with char_cols[1]:
                # Only count selected characters
                char_removed = sum(v for cls_id, v in delta.items()
                                 if v is not None and v < 0
                                 and CLS_MAP[int(cls_id) if isinstance(cls_id, str) else cls_id] in selected_classes)
                st.metric("Chars Removed", f"{char_removed:,}")
        else:
            st.info("No changes in this run compared to previous run.")
    else:
        st.info("📊 Need at least 2 runs to show delta counts.")