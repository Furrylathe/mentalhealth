# AI Mental Health Companion 🧠

## Description

This project provides a user-friendly interface for interacting with a mental health care AI system built using Streamlit and LangChain. It leverages the power of Google Generative AI (GenAI) to offer supportive and actionable advice based on user-provided questions.

## Key Features

- **Mental Health Focus**: The AI focuses on mental health topics, ensuring responses are relevant and appropriate.
- **Conversational Interface**: Users can interact with the AI through a chat-like interface, fostering a natural and engaging experience.
- **Contextual Awareness**: The AI considers the context of the conversation when generating responses, providing more tailored guidance.
- **Streamlit Integration**: Streamlit simplifies deployment and user interaction, making the AI accessible through a web browser.
- **LangChain Utilization**: LangChain facilitates the AI workflow, including question-answering chain creation and prompt generation.

## Project Structure

- `ingest.py`: Preprocesses data for AI use (PDF, CSV, JSON parsing, text chunking, vector store creation).
- `app.py`: Streamlit application logic, manages user interaction, loads data, and calls the conversational chain.
- `requirements.txt`: Lists all necessary Python dependencies for project execution.
- `README.md`: This file (you're reading it now!).

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/ai-mental-health-companion.git
   cd ai-mental-health-companion
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # macOS/Linux: source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**

## Usage

1. **Navigate to the project directory in your terminal.**
2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
3. **Access the app in your web browser at http://localhost:8501.**

## Dependencies

- Streamlit
- LangChain
- LangChain Google GenAI
- LangChain Community Vector Stores (FAISS)
- dotenv
- PyPDF2 (if processing PDFs)
- pandas (if processing CSVs)

## Contributing

- Fork the repository.
- Create a feature branch for your changes.
- Write clear and concise commit messages.
- Submit a pull request for review and merging.
