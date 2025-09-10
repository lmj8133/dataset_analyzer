"""YOLO11 License Plate Character Recognition Dashboard."""

import streamlit as st
from datetime import datetime
from core.data_manager import data_manager
from ui_pages.gt import render_gt_page
from ui_pages.add_run import render_add_run_page
from ui_pages.trends import render_trends_page

st.set_page_config(
    page_title="YOLO11 Recognition Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables."""
    if 'gt_labels' not in st.session_state:
        st.session_state.gt_labels = {}
    if 'gt_slim' not in st.session_state:
        st.session_state.gt_slim = {}
    if 'runs' not in st.session_state:
        st.session_state.runs = []
    if 'auto_save_enabled' not in st.session_state:
        st.session_state.auto_save_enabled = True
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = None
    if 'session_loaded' not in st.session_state:
        st.session_state.session_loaded = False

def main():
    """Main application."""
    init_session_state()
    
    # Auto-load on first run
    if not st.session_state.session_loaded:
        session_info = data_manager.get_session_info()
        if session_info.get("exists") and session_info.get("gt_count", 0) > 0:
            # Show option to load previous session
            with st.sidebar:
                if st.button("📥 Restore previous session", type="primary", use_container_width=True):
                    success, message = data_manager.load_session()
                    if success:
                        st.success(message)
                        st.session_state.session_loaded = True
                        st.rerun()
    
    st.sidebar.title("🚗 YOLO11 Dashboard")
    st.sidebar.markdown("License Plate Character Recognition")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Navigation",
        ["📁 GT Upload", "➕ Add Run", "📈 Trends"],
        help="Select a page to navigate"
    )
    
    st.sidebar.divider()
    
    # Data Persistence Section
    with st.sidebar.expander("💾 Data Persistence", expanded=False):
        # Session info
        session_info = data_manager.get_session_info()
        
        if session_info.get("exists"):
            st.success("📁 Saved session found")
            if "last_modified" in session_info:
                st.caption(f"Modified: {session_info['last_modified'].strftime('%Y-%m-%d %H:%M')}")
            if "gt_count" in session_info:
                st.caption(f"GT: {session_info['gt_count']} | Runs: {session_info['run_count']}")
        else:
            st.info("No saved session")
        
        col1, col2 = st.columns(2)
        
        # Save button
        with col1:
            if st.button("💾 Save", use_container_width=True):
                success, message = data_manager.save_session()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Load button
        with col2:
            if st.button("📂 Load", use_container_width=True):
                success, message = data_manager.load_session()
                if success:
                    st.success(message)
                    st.session_state.session_loaded = True
                    st.rerun()
                else:
                    st.error(message)
        
        # Auto-save option
        st.session_state.auto_save_enabled = st.checkbox(
            "Auto-save on changes",
            value=st.session_state.auto_save_enabled,
            help="Automatically save after adding GT or runs"
        )
        
        # Last save time
        if st.session_state.last_save_time:
            st.caption(f"Last saved: {st.session_state.last_save_time.strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Export/Import
        st.markdown("**Export/Import**")
        
        # Export button
        export_data = data_manager.export_data()
        if export_data:
            st.download_button(
                label="📥 Export Data",
                data=export_data,
                file_name=f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz",
                mime="application/gzip",
                use_container_width=True
            )
        
        # Import file uploader
        uploaded_file = st.file_uploader(
            "📤 Import Data",
            type=["json", "gz"],
            key="import_uploader",
            help="Upload exported session file"
        )
        
        if uploaded_file:
            file_data = uploaded_file.read()
            success, message = data_manager.import_data(file_data)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    st.sidebar.divider()
    
    with st.sidebar:
        st.markdown("### 📊 Current Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gt_count = len(st.session_state.gt_labels)
            st.metric("GT Plates", gt_count)
        
        with col2:
            run_count = len(st.session_state.runs)
            st.metric("Runs", run_count)
        
        if st.session_state.runs:
            latest_run = st.session_state.runs[-1]
            st.markdown(f"**Latest:** {latest_run['name']}")
            st.caption(f"Plate Acc: {latest_run['metrics']['plate_accuracy']:.2%}")
            st.caption(f"Char Acc: {latest_run['metrics']['char_accuracy']:.2%}")
        
        st.divider()
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            **YOLO11 Recognition Dashboard**
            
            A minimal, fast dashboard for tracking license plate character recognition performance across training runs.
            
            - **String-based metrics** (no IoU/confidence)
            - **In-memory storage** (no persistence)
            - **Real-time trend analysis**
            
            Built with Streamlit + Plotly
            """)
        
        if st.button("🔄 Reset All Data", type="secondary", use_container_width=True):
            st.session_state.clear()
            init_session_state()
            data_manager.clear_all_data()
            st.success("All data cleared!")
            st.rerun()
    
    if page == "📁 GT Upload":
        render_gt_page()
    elif page == "➕ Add Run":
        render_add_run_page()
    elif page == "📈 Trends":
        render_trends_page()

if __name__ == "__main__":
    main()