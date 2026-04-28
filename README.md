# AI learning coach project

Description: An AI learning coach that helps students understand course material by retrieving relevant content from their notes and adapting explanations over a multi-turn conversation.

The notebooks do not render on Github, so read all the code in the .py files in src

## What it does

The AI Learning Coach is a retrieval-augmented generation system that allows students to upload their own course notes and receive accurate, context-grounded answers to their questions through a multi-turn conversational interface. The system works by semantically chunking the uploaded notes, embedding them into a FAISS vector index, and using a two-stage retrieval pipeline of dense cosine similarity search followed by cross-encoder reranking to find the most relevant passages for any given question. These retrieved chunks are injected into a chain-of-thought prompt alongside few-shot examples, and passed to a LoRA fine-tuned TinyLlama model that generates a focused, student-friendly explanation. The conversation manager tracks the full dialogue history with token-aware truncation, allowing the model to handle follow-up questions that depend on previous turns, while a Gradio interface displays both the model's answers and the source chunks that informed them, giving students full transparency into where the information came from.

## Quick Start

1. Clone the repository into your local device by typing in the terminal: git clone (ssh key)
2. Install the required packages by running (pip install -r requirements.txt) in the terminal
3. Upload class notes (.txt or .pdf) into data folder

Otherwise, the best way to run the project is download the ipynb notebooks and run them on Google Colab with the CUDA GPU. In each notebook, select runtime and change runtime type to T4 GPU before running

Upload class notes (.txt or .pdf) to your Google Drive at MyDrive/372-Project/data/

Open the notebooks folder and run the files in this order:
1. finetune_lora.ipynb
2. rag_pipeline.ipynb
3. chat_system.ipynb

After running the last notebook, a user interface will be output that allows the student to submit questions during study sessions and learn from the AI's output.

## Video Links

## Evaluation

Steps, training loss, and validation loss from finetuning the Tinyllama model on the alpaca dataset:
<img width="650" height="556" alt="image" src="https://github.com/user-attachments/assets/198e7317-3545-4f08-bbf9-89e34f3b7225" />

