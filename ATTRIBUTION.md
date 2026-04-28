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

Sources
https://huggingface.co/datasets/yahma/alpaca-cleaned

https://etica.uazuay.edu.ec/sites/etica.uazuay.edu.ec/files/public/Philosophy%20101_%20From%20Plato%20and%20Socrates%20to%20Ethics%20and%20Metaphysics%2C%20an%20Essential%20Primer%20on%20the%20History%20of%20Thought%20%28%20PDFDrive%20%29.pdf

### Libraries

PyTorch>=2.0.0Model training and inferenceHuggingFace Transformers>=4.40.0Model loading, Trainer, tokenizationPEFT>=0.10.0LoRA configuration and fine-tuningAccelerate>=0.30.0Distributed training utilitiesbitsandbytes>=0.43.0Mixed precision and quantizationDatasets>=2.19.0Dataset loading and preprocessingsentence-transformers>=2.7.0Embedding and cross-encoder modelsFAISS (faiss-cpu)>=1.8.0Vector similarity search indexevaluate>=0.4.0ROUGE-L metric computationrouge-score>=0.1.2ROUGE scoring backendbert-score>=0.3.13BERTScore semantic similarity metricPyMuPDF (fitz)>=1.24.0PDF text extractionGradio>=4.0.0Chat UINumPy>=1.24.0Numerical operationsMatplotlib>=3.7.0Training curve visualization
