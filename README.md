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

Demo video link: https://duke.zoom.us/rec/share/xcEn6Bjn-8piGb8dahZ9VenerXGen9XZ3EjLHzmzSC-3JnQAOp5EO7wEc189SBY.BLsV4SaD5p2KSA14?startTime=1777422845000 

Technical video: https://duke.zoom.us/rec/share/vTam05lWGPJ4X1eg783E-hhyFO9_UafNvcK_zcEzx4CrrwGwEfJxqFPJQdl3c7J_.isUawWqx8RjXqur6?startTime=1777446215000


## Evaluation

Steps, training loss, and validation loss from finetuning the Tinyllama model on the alpaca dataset:
<img width="650" height="556" alt="image" src="https://github.com/user-attachments/assets/198e7317-3545-4f08-bbf9-89e34f3b7225" />

<img width="2564" height="890" alt="image" src="https://github.com/user-attachments/assets/29a0b690-319d-4b00-ba02-2c6b62f16aca" />
As we can see, the training loss drastically decreases in the first few training steps 0-200, but from 200-800 steps it fluctuates while still following a general trend of decline in loss. The validation loss, on the other hand, follows a smoother and more consistent decline over steps, showing the improvement of the model and increased accuracy over training.



After testing the finetuned llama model with the prompt style and inputing "Explain the concept of gradient descent to a student who is learning for the first time." the output is:

"Gradient descent is a popular algorithm used in machine learning and optimization. It works by finding the direction of least curvature, which is a tangent line to a curve at a given point, towards the point where the curve changes direction. The algorithm starts from an initial guess and iteratively updates the current estimate until it converges or reaches a local minimum. Gradient descent has several advantages over other methods such as stochastic gradient descent (SGD), where the update rule involves averaging gradients. One of the main advantages of gradient descent is its ability to handle non-convex problems, while SGD may not converge on some cases. Additionally, gradient descent can be used in situations where there are no initial estimates or when the data set is large, making it more efficient than other algorithms. In summary, gradient descent is an effective method for finding local minima or maxima in functions that have a smooth curvature, and it is easy to implement in many programming languages. It is one of the most widely used algorithms in machine learning and optimization. Students new to machine learning should learn about gradient descent and the related concepts before moving on to more advanced topics."

The output is on topic and grammatically correct, which is good, with display of temperature sampling. However, it's factually imprecise with some incorrect information, such as saying that gradient descent is the direction of least curvature, which is incorrect. This is probably due to the model hallucinating from limited training.
This is our iteration 1 baseline model, and I will implement RAG in the next notebook to improve factual precision and correctness of the outputs to give students accurate information when using this model and reduce hallucinations.

In rag_pipeline.ipynb, when trying to find the top 3 chunks most similar to the example query using cosine similarity "Who was Thomas Hobbes", the output is:

Top 3 dense results:
  [126] score=0.639 | THOMAS HOBBES (1588–1679) A new philosophical system
Thomas Hobbes was born on A...
  [127] score=0.561 | monarchy, Hobbes did not believe the king’s right to rule was from God; rather,
...
  [128] score=0.499 | the mechanical processes of the nervous system). As such, Hobbes claimed that
hu...

Then when trying to use a reranker on top of the cosine similarity, the output is:

Query: Who was Thomas Hobbes?

  [1] rerank_score=7.651
  THOMAS HOBBES (1588–1679) A new philosophical system
Thomas Hobbes was born on April 5, 1588, in Malmesbury, England. Though
his father disappeared when he was young, Hobbes’s uncle paid for his
educa...

  [2] rerank_score=3.307
  monarchy, Hobbes did not believe the king’s right to rule was from God; rather,
it was a social contract agreed upon by the people.
Hobbes was convinced that there needed to be an overhaul of philosop...

  [3] rerank_score=1.058
  the mechanical processes of the nervous system). As such, Hobbes claimed that
humans avoid pain and pursue pleasure in an effort to seek out our own self-
interest (which makes humans’ judgment extrem...

When generating text with few-shot, the output is:

Instruction:
You are a helpful learning coach. Here are examples of good answers:

Q: What is the ship of Theseus?
A: The Ship of Theseus is a philosophical thought experiment that askswhether an object remains the same if all of its original parts aregradually replaced. It questions the nature of identity and whethercontinuity comes from physical components or from history and perception.

Q: What is existentialism?
A: Existentialism is a philosophical movement that focuses on individualfreedom, choice, and responsibility, arguing that people create meaningthrough their actions rather than discovering a pre-given purpose.

Now answer this question using only the notes below.

 Input:
[Source 1]
HEDONISM It’s all about pleasure and pain
The term hedonism actually refers to several theories that, while different from
one another, all share the same underlying notion: Pleasure and pain are the only
important elements of the specific phenomena the theories describe. In
philosophy, hedonism is often discussed as a theory of value. This means that
pleasure is the only thing intrinsically valuable to a person at all times and pain
is the only thing that is intrinsically not valuable to an individual. To hedonists,
the meaning of pleasure and pain is broad so that it can relate to both mental and
physical phenomena.
ORIGINS AND HISTORY OF HEDONISM
The first major hedonistic movement dates back to the fourth century b.c. with
the Cyrenaics, a school of thought founded by Aristippus of Cyrene. The
Cyrenaics emphasized Socrates’ belief that happiness is one of the results of
moral action, but also believed that virtue had no intrinsic value. They believed
that pleasure, specifically physical pleasure over mental pleasure, was the
ultimate good and that immediate gratification was more desirable than having
to wait a long time for pleasure.
Following the Cyrenaics was Epicureanism (led by Epicurus), which was a
form of hedonism quite different from that of Aristippus. While he agreed that
pleasure was the ultimate good, Epicurus believed that pleasure was attained
through tranquility and a reduction of desire instead of immediate gratification.
According to Epicurus, living a simple life full of friends and philosophical
discussion was the highest pleasure that could be attained.

[Source 2]
During the Middle Ages, hedonism was rejected by Christian philosophers
because it did not mesh with Christian virtues and ideals, such as faith, hope,
avoiding sin, and helping others. Still, some philosophers argued hedonism had
its merits because it was God’s desire that people be happy.
Hedonism was most popular in the eighteenth and nineteenth centuries due to
the work of Jeremy Bentham and John Stuart Mill, who both argued for
variations of prudential hedonism, hedonistic utilitarianism, and motivational
hedonism.
VALUE AND PRUDENTIAL HEDONISM
In philosophy, hedonism usually refers to value and well-being. Value hedonism
states that pleasure is the only thing that is intrinsically valuable, while pain is
the only thing that is intrinsically invaluable.
Philosophical Definitions
INTRINSICALLY VALUABLE: The word intrinsically is thrown
around a lot when discussing hedonism, and it is a very important
word to understand. Unlike the word instrumental, use of the word
intrinsically implies that something is valuable on its own. Money is
instrumentally valuable. Having money only has real value when you
purchase something with it. Therefore, it is not intrinsically valuable.
Pleasure, on the other hand, is intrinsically valuable. When a person
experiences pleasure, even if it does not lead to something else, the
initial pleasure itself is enjoyable.

[Source 3]
According to value hedonism, everything that is of value is reduced to
pleasure. Based on this information, prudential hedonism then goes one step
further and claims that all pleasure, and only pleasure, can make an individual’s
life better, and that all pain, and only pain, can make an individual’s life worse.
PSYCHOLOGICAL HEDONISM
Psychological hedonism is a branch of hedonism that focuses on the
psychological aspects of pleasure. According to psychological hedonism,
pleasure is not just a physical experience, but also involves emotions,
thoughts, and feelings.
Psychological hedonism states that pleasure is not just a physical
experience, but also involves emotions, thoughts, and feelings. For example,
when someone experiences pleasure, they may feel happy, content, or
enjoyed. These emotions are what make pleasure pleasurable.
Psychological hedonism also suggests that pleasure is not just a physical
experience, but also involves thoughts and feelings. For instance, when
someone experiences pleasure, they may think happy thoughts, or feel
contented thoughts. These thoughts and feelings are what make pleasure
pleasurable.

[Source 4]
In summary, hedonism is a philosophical movement that focuses on the
meaning of pleasure and pain, and how these things relate to each other.
It is a movement that emphasizes the importance of pleasure and
desire, and the idea that pleasure is intrinsically valuable. However,
hedonism also recognizes the role of emotion, thought, and feeling in
determining the meaning of pleasure.

Input:
[Source 5]
HEDONISM It’s all about pleasure and pain
The term hedonism actually refers to several theories that, while different from
one another, all share the same underlying notion: Pleasure and pain are the only
important elements of the specific phenomena the theories describe. In philosophy, hedonism is often discussed as a theory of value. This means that pleasure is the only thing intrinsically valuable to a person at all times and pain is the only thing that is intrinsically not valuable to an individual. To hedonists, the meaning of pleasure and pain is broad so that it can relate to both mental and physical phenomena.
ORIGINS AND HISTORY OF HEDONISM
The first major hedonistic movement dates back to the fourth century b.c. With the Cyrenaics, a school of thought founded by Aristippus of Cyrene. The Cyrenaics emphasized Socrates’ belief that happiness is one of the results of moral action, but also believed that pleasure, specifically physical pleasure over mental pleasure, was the ultimate good and that immediate gratification was more desirable than having to wait a long time for pleasure.
Following the Cyrenaics was Epicureanism (led by Epicurus), which was a form of hedonism quite different from that of Aristippus. While he agreed that pleasure was the ultimate good, Epicurus believed that living a simple life full of friends and philosophical discussion was the highest pleasure that could be attained.
According to value hedonism, everything that is of value is reduced to pleasure. Based on this information, prudential hedonism then goes one step further and claims that all pleasure, and only pleasure, can make an individual’s life better, and that all pain, and only pain, can make an individual’s life worse. PSYCHOLOGICAL HEDONISM
Psychological hedonism is a branch of hedonism that focuses on the psychological aspects of pleasure. According to psychological hedonism, pleasure is not just a physical experience, but also involves emotions, thoughts, and feelings. PERSONALITY HEDONISM
Personality hedonism is a sub-branch

This shows how the model is still hallucinating sources instead of giving an answer and repeating the prompt, showing its errors and room for improvement.

Then when prompt engineering with the prompt styles zero-shot, few-shot, and cot, the rouge scores are in the table as follows:
<img width="1266" height="192" alt="image" src="https://github.com/user-attachments/assets/7aa3f7d0-4faf-463e-addd-db5bb8eafedd" />

Then when iterating 2 model improvements with iteration 1 being only using cosine similarity for retrieval and iteration 2 being using cosine retrieval plus a reranker, the rouge scores and graphs are output as follows:

<img width="1802" height="1208" alt="image" src="https://github.com/user-attachments/assets/ad6cbf54-a005-48fb-ba99-6bbdd813cc5e" />

As we can see, rouge scores improved with iteration 2, and adding a reranker improved generation

Then when testing the model with Rouge scores, BERTscore, and Faithful scores, the output is as follows:

<img width="986" height="136" alt="image" src="https://github.com/user-attachments/assets/4ecb3f6b-76d7-4bfb-988f-477f9fddea1a" />





