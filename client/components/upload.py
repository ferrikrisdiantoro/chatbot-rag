import streamlit as st
from utils.api import upload_pdfs_api
import time

def render_uploader():
    # Custom CSS untuk sidebar styling
    st.markdown("""
    <style>
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar content styling */
    .sidebar .sidebar-content {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: #667eea;
    }
    
    /* File info styling */
    .file-info {
        background: #f1f3f4;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    
    /* Success/Error message styling */
    .success-msg {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .error-msg {
        background: linear-gradient(90deg, #f44336, #da190b);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.7rem 1rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:disabled {
        background: #cccccc;
        transform: none;
        box-shadow: none;
    }
    
    /* Clear button styling */
    .clear-btn {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%) !important;
    }
    
    /* File counter styling */
    .file-counter {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
    if "upload_progress" not in st.session_state:
        st.session_state.upload_progress = 0
    
    # Sidebar header dengan icon
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem;'>
        <h2 style='color: #667eea; margin: 0;'>📄 Upload PDFs</h2>
        <p style='color: #666; font-size: 0.9rem; margin: 0.5rem 0;'>
            Upload dokumen PDF untuk analisis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader dengan styling yang lebih baik
    uploaded_files = st.sidebar.file_uploader(
        "Pilih file PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Anda dapat memilih beberapa file PDF sekaligus"
    )
    
    # Display file information
    if uploaded_files:
        # File counter
        st.sidebar.markdown(f"""
        <div class='file-counter'>
            📁 {len(uploaded_files)} file terpilih
        </div>
        """, unsafe_allow_html=True)
        
        # File details
        st.sidebar.markdown("**📋 Detail File:**")
        total_size = 0
        
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / (1024 * 1024)  # Convert to MB
            total_size += file_size
            
            st.sidebar.markdown(f"""
            <div class='file-info'>
                <strong>{i}.</strong> {file.name}<br>
                <small>📏 {file_size:.2f} MB</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Total size info
        st.sidebar.markdown(f"""
        <div style='text-align: center; margin: 1rem 0; padding: 0.5rem; 
                    background: #e3f2fd; border-radius: 8px; color: #1976d2;'>
            <strong>Total: {total_size:.2f} MB</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload and Clear buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            upload_clicked = st.button(
                "🚀 Upload",
                key="upload_btn",
                help="Upload file ke database",
                use_container_width=True
            )
        
        with col2:
            if st.button(
                "🗑️ Clear",
                key="clear_btn",
                help="Hapus semua file",
                use_container_width=True
            ):
                st.session_state.uploaded_files_info = []
                st.session_state.upload_status = None
                st.rerun()
        
        # Process upload
        if upload_clicked:
            st.session_state.upload_status = "uploading"
            
            # Progress bar
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            try:
                # Simulate progress (you can integrate with actual upload progress)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f'📤 Uploading... {i + 1}%')
                    time.sleep(0.01)  # Small delay for visual effect
                
                # Actual upload
                response = upload_pdfs_api(uploaded_files)
                
                if response.status_code == 200:
                    st.session_state.upload_status = "success"
                    st.session_state.uploaded_files_info = [
                        {"name": file.name, "size": f"{len(file.getvalue()) / (1024 * 1024):.2f} MB"}
                        for file in uploaded_files
                    ]
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Success message
                    st.sidebar.markdown("""
                    <div class='success-msg'>
                        ✅ Upload berhasil!<br>
                        <small>File siap untuk dianalisis</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-refresh after success
                    time.sleep(2)
                    st.rerun()
                    
                else:
                    st.session_state.upload_status = "error"
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Error message
                    st.sidebar.markdown(f"""
                    <div class='error-msg'>
                        ❌ Upload gagal!<br>
                        <small>{response.text}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.session_state.upload_status = "error"
                progress_bar.empty()
                status_text.empty()
                
                # Error message
                st.sidebar.markdown(f"""
                <div class='error-msg'>
                    ❌ Terjadi kesalahan!<br>
                    <small>{str(e)}</small>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Empty state
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; color: #666; 
                    border: 2px dashed #ddd; border-radius: 10px; margin: 1rem 0;'>
            <h4 style='color: #999;'>📁 Belum ada file</h4>
            <p style='font-size: 0.9rem; margin: 0;'>
                Klik "Browse files" untuk memilih PDF
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display recently uploaded files info
    if st.session_state.uploaded_files_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📚 File Terakhir Diupload:**")
        
        for file_info in st.session_state.uploaded_files_info:
            st.sidebar.markdown(f"""
            <div class='file-info'>
                ✅ {file_info['name']}<br>
                <small>📏 {file_info['size']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;'>
        💡 <strong>Tips:</strong><br>
        • Format yang didukung: PDF<br>
        • Maksimal ukuran: 10MB per file<br>
        • Dapat upload multiple files<br>
        • File akan diproses otomatis
    </div>
    """, unsafe_allow_html=True)