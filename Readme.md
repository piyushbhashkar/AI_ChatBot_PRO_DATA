pipenv install streamlit langchain_groq langchain_community pypdf sentence-transformers
pipenv install langchain_text_splitters
pipenv install langchain_community
pip install unstructured python-pptx
pip install "unstructured[docx]" python-docx

# Use the below python version to create new project 
pipenv --python 3.11
# How to Run the UI 
streamlit run dev.py