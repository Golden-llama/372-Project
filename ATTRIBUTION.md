### Use of AI

Claude version Sonnet 4.6 was used to generate code to produce the gradio user interface that allows students to directly interact with the model and ask questions and learn from generated answers.


### Pretrained Models

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 

Source: [HuggingFaceBase](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

Model: sentence-transformers/all-MiniLM-L6-v2

Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Model: cross-encoder/ms-marco-MiniLM-L-6-v2

Source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 

### Datasets

Sources:

https://huggingface.co/datasets/yahma/alpaca-cleaned

https://etica.uazuay.edu.ec/sites/etica.uazuay.edu.ec/files/public/Philosophy%20101_%20From%20Plato%20and%20Socrates%20to%20Ethics%20and%20Metaphysics%2C%20an%20Essential%20Primer%20on%20the%20History%20of%20Thought%20%28%20PDFDrive%20%29.pdf

### Libraries

PyTorch>=2.0.0

Transformers>=4.40.0

PEFT>=0.10.0

Accelerate>=0.30.0

bitsandbytes>=0.43.0

Datasets>=2.19.0

sentence-transformers>=2.7.0

FAISS (faiss-cpu)>=1.8.0

evaluate>=0.4.0

rouge-score>=0.1.2

bert-score>=0.3.13c

PyMuPDF (fitz)>=1.24.0

Gradio>=4.0.0Chat UINumPy>=1.24.0

Matplotlib>=3.7.0
