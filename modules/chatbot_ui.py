# modules/chatbot_ui.py

import streamlit as st

class ChatBotUI:
    def __init__(self, qa_pipeline, vector_store,selected_subject):
        self.qa_pipeline = qa_pipeline
        self.vector_store = vector_store
        self.selected_subject = selected_subject

    def render(self):
        #st.set_page_config(page_title="NEET Subject Chatbot", page_icon="📚", layout="centered")
        # Ensure this is the very first Streamlit command
        #st.set_page_config(page_title="NEET Subject Chatbot", page_icon="📚", layout="centered")

        # Custom styles
        st.markdown("""
            <style>
            .main {
                background-color: #f7f9fc;
                padding: 20px;
                border-radius: 10px;
            }
            .stTextInput>div>div>input {
                background-color:#000000;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 5px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Define layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title(f"🤖 NEET {self.selected_subject} Chatbot")
            st.write(f"Ask any concept from the CBSE/NEET {self.selected_subject} syllabus 📚")

            # Input to select the subject
            #selected_subject = st.selectbox("Choose your subject:", ["Physics", "Chemistry", "Biology"])

            user_question = st.text_input(f"💬 Ask a {self.selected_subject} question:")

            if user_question:
                with st.spinner("🔍 Searching for answers..."):
                    # Retrieve context based on the selected subject
                    context = self.vector_store.retrieve_context(user_question, subject=self.selected_subject)

                    if not context.strip():
                        st.warning("⚠️ No relevant context found. Try rephrasing your question.")
                    else:
                        # Get the answer using the QAPipeline
                        answer = self.qa_pipeline.get_answer(user_question, context)
                        st.success("✅ Answer generated!")

                        st.subheader("📘 Answer:")
                        st.write(answer)

                        if st.checkbox("📂 Show context used for answer"):
                            st.code(context, language="markdown")

        # Sidebar content for usage instructions
        st.sidebar.title("🛠️ How to Use")
        st.sidebar.markdown("""
        - Enter a **clear, subject-related** question for the selected subject.
        - Use **scientific terms** for better accuracy.
        - Example:  
          - ✅ *What is photosynthesis?*  
          - ✅ *Explain the laws of motion in Physics.*

        ---

        🧪 *This chatbot is powered by Transformers & Vector DB.* 
        """)
