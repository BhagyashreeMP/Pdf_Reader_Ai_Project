# PDF Q&A Chatbot

This project implements a simple PDF question-and-answer chatbot using Gradio and a BERT-based model for natural language processing. Users can upload a PDF document and ask questions about its content.

## Features

- Upload PDF files to extract text.
- Ask questions related to the content of the PDF.
- Get answers using a pre-trained BERT model from Hugging Face.

## Technologies Used

- **Gradio**: For creating the user interface.
- **PyMuPDF (fitz)**: For extracting text from PDF files.
- **Transformers (Hugging Face)**: For utilizing a pre-trained BERT model for question answering.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   
2. Install the required libraries
 ## Code Overview

### Model Setup
The chatbot uses a pre-trained BERT model, specifically `"bert-large-uncased-whole-word-masking-finetuned-squad"`, which is designed for answering questions based on plain text.

### Text Extraction
The `extract_text_from_pdf(pdf_file)` function reads and extracts text from the uploaded PDF file using PyMuPDF. It processes each page to retrieve the text content.

### Question Answering
The `answer_question(question, pdf_file)` function takes a user's question and the uploaded PDF, extracts the text, and utilizes the Hugging Face question-answering pipeline to provide an answer. If no text is found in the PDF, it returns an error message.

### Gradio Interface
A user-friendly Gradio interface is created with two inputs:
- A text box for asking questions.
- A file upload option for the PDF.

The `answer_question()` function processes the user inputs and provides answers. The interface is launched in debug mode for easier troubleshooting.

