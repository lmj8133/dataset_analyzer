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
                placeholder=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Leave empty for auto-generated name"
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
                
                if not run_name:
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
                    
                    metrics = evaluate_run(
                        st.session_state.gt_labels,
                        pred_labels,
                        train_counts,
                        CLS_MAP
                    )
                    
                    run_data = {
                        'name': run_name,
                        'description': run_description,
                        'timestamp': datetime.now(),
                        'metrics': metrics,
                        'train_counts': train_counts
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
        st.subheader("📊 Run Summary")
        
        df_runs = pd.DataFrame([
            {
                'Run': run['name'],
                'Time': run['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'EMR': f"{run['metrics']['emr']:.2%}",
                'Char Acc': f"{run['metrics']['char_accuracy']:.2%}",
                'Images': run['metrics']['n_images'],
                'Total Chars': run['metrics']['total_gt_chars'],
                'Edit Dist': run['metrics']['total_edit_distance'],
                'Description': run['description'][:50] + '...' if len(run['description']) > 50 else run['description']
            }
            for run in st.session_state.runs
        ])
        
        st.dataframe(
            df_runs,
            use_container_width=True,
            hide_index=True
        )
        
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