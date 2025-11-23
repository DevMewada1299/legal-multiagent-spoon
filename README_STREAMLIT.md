# Legal Document Analyzer - Streamlit App

## Overview
This Streamlit app provides a user-friendly interface for analyzing legal documents using the legal-multiagent pipeline.

## Features
- 📄 PDF file upload
- 🏷️ Automatic clause classification
- 🔍 Obligation extraction
- 🔎 Semantic clause retrieval
- 📝 AI-powered document summarization

## Installation

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Running the App

Launch the Streamlit app with:
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser (typically at `http://localhost:8501`).

## Usage

1. **Upload PDF**: Click "Browse files" and select a legal document (PDF format)
2. **Analyze**: Click the "Analyze Document" button to process the document
3. **View Results**: Navigate through the three tabs to view:
   - **Classified Clauses**: All extracted clauses with labels and obligation flags
   - **Retrieved Clauses**: Clauses relevant to termination conditions
   - **Summary**: AI-generated summary of the entire document

## Environment Variables

Make sure your `.env` file contains the necessary API keys:
- Spoon AI SDK credentials
- Google Gemini API key (for summarization)

## Notes
- The app processes documents asynchronously for better performance
- Temporary files are automatically cleaned up after processing
- Large documents may take a few minutes to process
