### Installation instructions

1. Clone the repository into your local device by typing in the terminal: git clone (ssh key)
2. Install the required packages by running (pip install -r requirements.txt) in the terminal
3. Upload class notes (.txt or .pdf) into data folder

Otherwise, the best way to run the project is download the ipynb notebooks and run them on Google Colab with the CUDA GPU. In each notebook, select runtime and change runtime type to T4 GPU before running

Upload class notes (.txt or .pdf) to your Google Drive at MyDrive/372-Project/data/

Open the notebooks folder and run the files in this order:

finetune_lora.ipynb
rag_pipeline.ipynb
chat_system.ipynb

After running the last notebook, a user interface will be output that allows the student to submit questions during study sessions and learn from the AI's output.

Using the Chat Interface:

1. Open the Gradio share link from the last cell of chat_system.ipynb
2. Type a question about your course notes in the text box
3. Press Send or hit Enter
4. View the answer in the chat panel and retrieved source chunks on the right
5. Ask follow-up questions since the coach remembers the conversation
6. Use Clear Conversation to start a fresh session
