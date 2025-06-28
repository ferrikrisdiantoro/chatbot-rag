import streamlit as st
from utils.api import ask_question
from audio.stt import recognize_speech_from_mic
from audio.tts import speak_text

def render_chat():
    # Custom CSS untuk styling yang lebih baik
    st.markdown("""
    <style>
    /* Hide default streamlit elements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Chat container styling */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    /* Input container styling */
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Mic button specific styling */
    .mic-button {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%) !important;
    }
    
    /* TTS button styling */
    .tts-button {
        background: linear-gradient(90deg, #4ecdc4 0%, #44a08d 100%) !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.8rem !important;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f1f3f4;
        color: #333;
        padding: 0.8rem 1rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        margin-right: 2rem;
    }
    
    /* Toast styling */
    .stToast {
        background: linear-gradient(90deg, #4ecdc4 0%, #44a08d 100%);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header dengan emoji dan styling
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #667eea; margin: 0;'>💬 Chat with Your Documents</h2>
        <p style='color: #666; margin: 0.5rem 0;'>Ask questions about your uploaded documents</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "mic_input" not in st.session_state:
        st.session_state.mic_input = ""
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    # TTS button untuk assistant messages
                    if msg["role"] == "assistant":
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button(
                                "🔊", 
                                key=f"tts_{i}",
                                help="Bacakan jawaban ini",
                                use_container_width=True
                            ):
                                with st.spinner("🗣️ Sedang berbicara..."):
                                    speak_text(msg["content"])
                                    st.success("✅ Selesai berbicara!")
            
            # Show processing indicator jika AI sedang berpikir
            if st.session_state.processing:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Sedang berpikir..."):
                        st.empty()  # Placeholder untuk menunjukkan AI sedang memproses
        else:
            # Welcome message
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #666;'>
                <h4>👋 Selamat datang!</h4>
                <p>Mulai percakapan dengan mengetik pertanyaan atau menggunakan mikrofon</p>
            </div>
            """, unsafe_allow_html=True)

    # Input section dengan layout yang lebih baik
    st.markdown("---")
    
    # Create columns for input layout
    input_col1, input_col2 = st.columns([6, 1])
    
    with input_col1:
        user_input = st.chat_input("💭 Ketik pertanyaan Anda di sini...")
    
    with input_col2:
        mic_clicked = st.button(
            "🎤", 
            key="mic_button",
            help="Klik untuk berbicara",
            use_container_width=True
        )

    # Handle microphone input
    if mic_clicked:
        with st.spinner("🎧 Mendengarkan..."):
            try:
                recognized = recognize_speech_from_mic()
                if recognized:
                    st.session_state.mic_input = recognized
                    st.toast(f"🗣️ Terdeteksi: '{recognized}'", icon="🎉")
                    # Auto-submit the recognized text
                    user_input = recognized
                else:
                    st.toast("❌ Tidak ada suara yang terdeteksi", icon="⚠️")
            except Exception as e:
                st.toast(f"❌ Error: {str(e)}", icon="🚨")

    # Process user input
    if user_input and not st.session_state.processing:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Set processing flag
        st.session_state.processing = True
        
        # Clear mic input
        if "mic_input" in st.session_state:
            st.session_state.mic_input = ""
        
        # Rerun to show user message immediately
        st.rerun()

    # Process AI response jika sedang processing
    if st.session_state.processing:
        # Get the last user message
        last_user_message = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
        
        if last_user_message:
            try:
                response = ask_question(last_user_message)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["response"]
                    sources = data.get("sources", [])
                    
                    # Format answer dengan sources jika ada
                    full_answer = answer
                    if sources:
                        full_answer += "\n\n**📄 Sumber:**\n"
                        for i, src in enumerate(sources, 1):
                            full_answer += f"{i}. `{src}`\n"
                    
                    # Add assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_answer
                    })
                    
                else:
                    error_msg = f"❌ Terjadi kesalahan: {response.text}"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"❌ Error sistem: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
            
            # Reset processing flag
            st.session_state.processing = False
            
            # Rerun to show assistant response
            st.rerun()

    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;'>
        💡 <strong>Tips:</strong> Gunakan mikrofon untuk input suara atau ketik langsung. 
        Klik tombol 🔊 untuk mendengar jawaban.
    </div>
    """, unsafe_allow_html=True)