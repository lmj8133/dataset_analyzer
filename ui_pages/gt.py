"""Ground Truth upload page."""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from core.data_manager import data_manager
from core.io_yolo import (
    process_label_upload, 
    process_folder_upload, 
    get_class_counts,
    compute_slim_format,
    reconstruct_string,
    CLS_MAP
)


def render_gt_page():
    """Render the GT upload page."""
    st.title("📁 Ground Truth Upload")
    st.markdown("Upload the fixed test-set GT labels (YOLO format)")
    
    # Initialize processing flag
    if 'processing_gt' not in st.session_state:
        st.session_state.processing_gt = False
    
    if 'gt_labels' in st.session_state and st.session_state.gt_labels:
        st.success(f"✅ GT labels loaded: {len(st.session_state.gt_labels)} images")
        
        if st.button("🔄 Reset GT Labels", type="secondary"):
            st.session_state.gt_labels = {}
            st.session_state.gt_slim = {}
            st.session_state.processing_gt = False
            st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Options")
        upload_type = st.radio(
            "Select upload type:",
            ["ZIP file", "Multiple files"],
            help="Upload as a ZIP archive or select multiple .txt files"
        )
        
        # Debug mode checkbox
        debug_mode = st.checkbox("🐛 Debug mode", help="Show detailed processing information")
    
    with col2:
        st.subheader("Upload Files")
        
        if upload_type == "ZIP file":
            uploaded_file = st.file_uploader(
                "Choose a ZIP file",
                type=['zip'],
                help="ZIP file containing .txt label files",
                key="zip_uploader"
            )
            
            if uploaded_file is not None and not st.session_state.processing_gt:
                st.session_state.processing_gt = True
                
                # Create progress containers
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    error_container = st.empty()
                    
                    start_time = time.time()
                    status_text.text("Starting to process ZIP file...")
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    try:
                        # Process with progress callback
                        labels, errors = process_label_upload(uploaded_file, progress_callback=update_progress)
                        
                        if labels:
                            # Compute slim format
                            status_text.text("Finalizing data structure...")
                            st.session_state.gt_labels = labels
                            st.session_state.gt_slim = compute_slim_format(labels)
                            st.session_state.gt_upload_time = datetime.now()
                            
                            # Show results
                            elapsed_time = time.time() - start_time
                            progress_bar.progress(1.0)
                            status_text.success(f"✅ Loaded {len(labels)} label files in {elapsed_time:.2f} seconds")
                            
                            # Auto-save if enabled
                            if st.session_state.get('auto_save_enabled', True):
                                save_success, save_msg = data_manager.save_session(auto_save=True)
                                if save_success:
                                    st.sidebar.success(f"💾 {save_msg}")
                            
                            if errors and debug_mode:
                                error_container.warning(f"⚠️ {len(errors)} files had errors")
                                with error_container.expander("Show errors"):
                                    for error in errors[:10]:  # Show first 10 errors
                                        st.text(error)
                            
                            # Clear processing flag after short delay
                            time.sleep(1)
                            st.session_state.processing_gt = False
                            st.rerun()
                        else:
                            status_text.error("❌ No valid label files found")
                            if errors:
                                error_container.error(f"Errors: {errors[0]}")
                            st.session_state.processing_gt = False
                            
                    except Exception as e:
                        status_text.error(f"❌ Error processing file: {str(e)}")
                        st.session_state.processing_gt = False
                        if debug_mode:
                            st.exception(e)
        
        else:  # Multiple files
            uploaded_files = st.file_uploader(
                "Choose .txt files",
                type=['txt'],
                accept_multiple_files=True,
                help="Select multiple YOLO format .txt files",
                key="multi_uploader"
            )
            
            if uploaded_files and not st.session_state.processing_gt:
                st.session_state.processing_gt = True
                
                # Create progress containers
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    error_container = st.empty()
                    
                    start_time = time.time()
                    status_text.text(f"Processing {len(uploaded_files)} files...")
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    try:
                        # Process with progress callback
                        labels, errors = process_folder_upload(uploaded_files, progress_callback=update_progress)
                        
                        if labels:
                            # Compute slim format
                            status_text.text("Finalizing data structure...")
                            st.session_state.gt_labels = labels
                            st.session_state.gt_slim = compute_slim_format(labels)
                            st.session_state.gt_upload_time = datetime.now()
                            
                            # Show results
                            elapsed_time = time.time() - start_time
                            progress_bar.progress(1.0)
                            status_text.success(f"✅ Loaded {len(labels)} label files in {elapsed_time:.2f} seconds")
                            
                            # Auto-save if enabled
                            if st.session_state.get('auto_save_enabled', True):
                                save_success, save_msg = data_manager.save_session(auto_save=True)
                                if save_success:
                                    st.sidebar.success(f"💾 {save_msg}")
                            
                            if errors and debug_mode:
                                error_container.warning(f"⚠️ {len(errors)} files had errors")
                                with error_container.expander("Show errors"):
                                    for error in errors[:10]:
                                        st.text(error)
                            
                            # Clear processing flag after short delay
                            time.sleep(1)
                            st.session_state.processing_gt = False
                            st.rerun()
                        else:
                            status_text.error("❌ No valid label files found")
                            if errors:
                                error_container.error(f"Errors: {errors[0]}")
                            st.session_state.processing_gt = False
                            
                    except Exception as e:
                        status_text.error(f"❌ Error processing files: {str(e)}")
                        st.session_state.processing_gt = False
                        if debug_mode:
                            st.exception(e)
    
    # Statistics section
    if 'gt_labels' in st.session_state and st.session_state.gt_labels:
        st.divider()
        
        st.subheader("📊 GT Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        labels = st.session_state.gt_labels
        
        with col1:
            st.metric("Total Images", len(labels))
        
        with col2:
            total_chars = sum(len(data['chars']) for data in labels.values())
            st.metric("Total Characters", f"{total_chars:,}")
        
        with col3:
            avg_length = total_chars / len(labels) if labels else 0
            st.metric("Avg Plate Length", f"{avg_length:.1f}")
        
        with col4:
            # Memory usage estimate
            import sys
            mem_usage = sys.getsizeof(st.session_state.gt_labels) / (1024 * 1024)
            st.metric("Memory Usage", f"{mem_usage:.2f} MB")
        
        st.subheader("📈 Class Distribution")
        
        class_counts = get_class_counts(labels)
        
        df_counts = pd.DataFrame([
            {'Character': CLS_MAP[i], 'Count': count}
            for i, count in class_counts.items()
            if count > 0
        ])
        
        if not df_counts.empty:
            df_counts = df_counts.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(df_counts.set_index('Character')['Count'])
            
            with col2:
                st.dataframe(
                    df_counts,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
        
        with st.expander("🔍 Sample Preview (First 5 images)"):
            sample_items = list(labels.items())[:5]
            
            for img_name, data in sample_items:
                # Reconstruct string on demand
                if 'string' not in data:
                    data['string'] = reconstruct_string(list(zip(data['chars'], data['xs'])))
                
                st.markdown(f"**{img_name}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"String: {data['string']}")
                with col2:
                    st.text(f"Length: {len(data['string'])}")
                
                char_details = [f"{CLS_MAP.get(c, '?')}({c})" for c in data['chars']]
                st.caption(f"Classes: {', '.join(char_details)}")
                st.divider()