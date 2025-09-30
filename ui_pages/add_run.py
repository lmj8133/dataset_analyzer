"""Add Run page for uploading training runs."""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from core.data_manager import data_manager
from core.io_yolo import (
    process_folder_upload,
    get_class_counts,
    CLS_MAP
)
from core.metrics import evaluate_run


def render_add_run_page():
    """Render the Add Run page."""
    st.title("➕ Add Training Run")
    
    if 'gt_slim' not in st.session_state or not st.session_state.gt_slim:
        st.error("⚠️ Please upload GT labels first!")
        st.stop()
    
    if 'runs' not in st.session_state:
        st.session_state.runs = []
    
    # Debug mode checkbox (outside form)
    debug_mode = st.checkbox("🐛 Debug mode", help="Show detailed processing information")
    
    with st.form("add_run_form"):
        st.subheader("Run Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            run_name = st.text_input(
                "Run Name (optional)",
                placeholder="run_YYYYMMDD_HHMMSS",
                help="Leave empty for auto-generated name",
                key="run_name_input"
            )
        
        with col2:
            run_description = st.text_area(
                "Description (optional)",
                placeholder="Brief description of this run...",
                height=68
            )
        
        st.divider()
        
        st.subheader("Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Labels** (for class counts)")
            train_files = st.file_uploader(
                "Training labels",
                type=['txt'],
                accept_multiple_files=True,
                key="train_upload",
                help="Select multiple YOLO format .txt files"
            )
        
        with col2:
            st.markdown("**Test Predictions** (for scoring)")
            pred_files = st.file_uploader(
                "Predictions on test set",
                type=['txt'],
                accept_multiple_files=True,
                key="pred_upload",
                help="Select multiple prediction .txt files for the fixed test set"
            )
        
        submitted = st.form_submit_button("🚀 Add Run", type="primary", use_container_width=True)
        
        if submitted and train_files and pred_files:
            # Create progress containers
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_container = st.empty()
                
                start_time = time.time()
                
                if not run_name or run_name.strip() == "":
                    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                try:
                    # Process training labels with progress
                    status_text.text(f"Processing {len(train_files)} training files...")
                    
                    def update_train_progress(progress, message):
                        progress_bar.progress(progress * 0.45)  # First 45% for training files
                        status_text.text(f"Training: {message}")
                    
                    train_labels, train_errors = process_folder_upload(train_files, progress_callback=update_train_progress)
                    
                    # Process prediction files with progress
                    status_text.text(f"Processing {len(pred_files)} prediction files...")
                    
                    def update_pred_progress(progress, message):
                        progress_bar.progress(0.45 + progress * 0.45)  # Next 45% for pred files
                        status_text.text(f"Predictions: {message}")
                    
                    pred_labels, pred_errors = process_folder_upload(pred_files, progress_callback=update_pred_progress)
                    
                    # Final processing
                    progress_bar.progress(0.9)
                    status_text.text("Computing metrics...")
                    
                    train_counts = get_class_counts(train_labels)
                    n_train_plates = len(train_labels)
                    
                    metrics = evaluate_run(
                        st.session_state.gt_labels,
                        pred_labels,
                        train_counts,
                        CLS_MAP
                    )
                    
                    # Calculate training set statistics
                    n_train_chars = sum(train_counts.values())

                    run_data = {
                        'name': run_name,
                        'description': run_description,
                        'timestamp': datetime.now(),
                        'metrics': metrics,
                        'train_counts': train_counts,
                        'n_train_plates': n_train_plates,
                        'n_train_chars': n_train_chars
                    }
                    
                    st.session_state.runs.append(run_data)
                    
                    # Show results
                    elapsed_time = time.time() - start_time
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Run '{run_name}' added successfully in {elapsed_time:.2f} seconds!")
                    
                    # Show errors if any
                    all_errors = train_errors + pred_errors
                    if all_errors and debug_mode:
                        error_container.warning(f"⚠️ {len(all_errors)} files had errors")
                        with error_container.expander("Show errors"):
                            for error in all_errors[:10]:  # Show first 10 errors
                                st.text(error)
                    
                    # Auto-save if enabled
                    if st.session_state.get('auto_save_enabled', True):
                        save_success, save_msg = data_manager.save_session(auto_save=True)
                        if save_success:
                            st.sidebar.success(f"💾 {save_msg}")
                    
                    st.balloons()
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.error(f"❌ Error processing files: {str(e)}")
                    if debug_mode:
                        st.exception(e)
    
    if st.session_state.runs:
        st.divider()
        
        # Run Summary header with edit button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Run Summary")
        with col2:
            if 'edit_mode' not in st.session_state:
                st.session_state.edit_mode = False
            
            if st.session_state.edit_mode:
                if st.button("💾 Save & Exit Edit", type="primary", use_container_width=True):
                    # Save edited descriptions
                    if 'edited_descriptions' in st.session_state:
                        for idx, desc in enumerate(st.session_state.edited_descriptions):
                            st.session_state.runs[idx]['description'] = desc
                        
                        # Auto-save if enabled
                        if st.session_state.get('auto_save_enabled', True):
                            success, msg = data_manager.save_session(auto_save=True)
                            if success:
                                st.success("✅ Descriptions updated and saved")
                        
                        # Clear temporary storage
                        del st.session_state.edited_descriptions
                    
                    st.session_state.edit_mode = False
                    st.rerun()
            else:
                if st.button("✏️ Edit Descriptions", use_container_width=True):
                    st.session_state.edit_mode = True
                    st.rerun()
        
        # Prepare DataFrame
        df_runs = pd.DataFrame([
            {
                'Run': run['name'],
                'Time': run['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Plate Acc': f"{run['metrics']['plate_accuracy']:.2%}",
                'Char Acc': f"{run['metrics']['char_accuracy']:.2%}",
                'Plates': run['n_train_plates'],
                'Total Chars': run.get('n_train_chars', sum(run['train_counts'].values())),
                'Edit Dist': run['metrics']['total_edit_distance'],
                'Description': run['description']  # Full description for editing
            }
            for run in st.session_state.runs
        ])
        
        # Display table based on mode
        if st.session_state.edit_mode:
            # Edit mode: Use data_editor
            edited_df = st.data_editor(
                df_runs,
                column_config={
                    "Description": st.column_config.TextColumn(
                        "Description",
                        help="Edit descriptions here, then click Save & Exit Edit",
                        width="large",
                        max_chars=500
                    )
                },
                disabled=["Run", "Time", "Plate Acc", "Char Acc", "Plates", "Total Chars", "Edit Dist"],
                use_container_width=True,
                hide_index=True,
                key="runs_editor"
            )
            # Store edited descriptions temporarily
            st.session_state.edited_descriptions = edited_df['Description'].tolist()
        else:
            # View mode: Clean columns layout with expandable descriptions
            with st.container():
                # Add subtle styling
                st.markdown("""
                    <style>
                    .run-table { margin-top: 0.5rem; }
                    .metric-value { font-weight: 600; }
                    </style>
                """, unsafe_allow_html=True)
                
                # Table header
                header_cols = st.columns([2, 2, 0.8, 1, 0.8, 1, 1, 3.5])
                headers = ["Run Name", "Time", "Plate Acc", "Char Acc", "Plates", "Chars", "Edit", "Description"]
                for col, header in zip(header_cols, headers):
                    col.markdown(f"**{header}**")
                
                st.markdown("---")
                
                # Data rows
                for idx, run in enumerate(st.session_state.runs):
                    cols = st.columns([2, 2, 0.8, 1, 0.8, 1, 1, 3.5])
                    
                    # Run name with code style for clarity
                    cols[0].markdown(f"`{run['name']}`")
                    
                    # Time in compact format
                    cols[1].caption(run['timestamp'].strftime('%m/%d %H:%M'))
                    
                    # Metrics without color coding
                    cols[2].caption(f"{run['metrics']['plate_accuracy']:.1%}")
                    cols[3].caption(f"{run['metrics']['char_accuracy']:.1%}")
                    
                    # Simple numbers (from training set)
                    cols[4].caption(str(run['n_train_plates']))
                    cols[5].caption(str(run.get('n_train_chars', sum(run['train_counts'].values()))))
                    cols[6].caption(str(run['metrics']['total_edit_distance']))
                    
                    # Description with inline expander
                    desc = run['description']
                    if len(desc) > 40:
                        with cols[7].expander(f"{desc[:40]}... 📄", expanded=False):
                            st.text_area(
                                "",
                                value=desc,
                                height=100,
                                disabled=True,
                                key=f"view_desc_{idx}",
                                label_visibility="collapsed"
                            )
                    else:
                        cols[7].caption(desc if desc else "—")
                    
                    # Subtle separator between rows
                    if idx < len(st.session_state.runs) - 1:
                        st.markdown("")  # Small spacing
        
        with st.expander("🗑️ Manage Runs"):
            st.warning("⚠️ Delete runs with caution")
            
            run_to_delete = st.selectbox(
                "Select run to delete",
                options=[run['name'] for run in st.session_state.runs],
                index=None
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🗑️ Delete Selected", type="secondary", disabled=not run_to_delete):
                    st.session_state.runs = [
                        run for run in st.session_state.runs 
                        if run['name'] != run_to_delete
                    ]
                    st.success(f"Deleted run: {run_to_delete}")
                    
                    # Auto-save after deletion
                    if st.session_state.get('auto_save_enabled', True):
                        data_manager.save_session(auto_save=True)
                    
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete All Runs", type="secondary"):
                    st.session_state.runs = []
                    st.success("All runs deleted")
                    
                    # Auto-save after deletion
                    if st.session_state.get('auto_save_enabled', True):
                        data_manager.save_session(auto_save=True)
                    
                    st.rerun()
        
        if len(st.session_state.runs) >= 1:
            st.info("💡 You can now view trends! Go to the Trends page.")
    else:
        st.info("📝 No runs added yet. Upload your first training run above.")