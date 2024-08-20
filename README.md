DistilBERT is a smaller, faster, and lighter version of the BERT (Bidirectional Encoder Representations from Transformers) model, designed to retain much of BERT's accuracy while reducing the model's size and computational requirements. DistilBERT achieves this by using a technique called knowledge distillation, where a smaller "student" model learns to mimic the behavior of a larger "teacher" model (in this case, BERT).

Key Features of DistilBERT:

Reduced Size: DistilBERT has about 40% fewer parameters than BERT, making it faster and more efficient while still maintaining 97% of BERT's performance on many NLP tasks.
Architecture: DistilBERT retains the Transformer architecture, consisting of multiple layers of self-attention and feed-forward networks.
Pretrained Weights: Like BERT, DistilBERT is pretrained on a large corpus of text using masked language modeling, making it effective for various downstream tasks, including text classification.
DistilBERT for Text Classification
For the task of cyberbullying classification, we fine-tune DistilBERT, adapting it to our specific needs:

Input Tokenization:

Tokenization: The input text is tokenized using the DistilBERT tokenizer, which converts text into tokens that correspond to the vocabulary used during pretraining.
Padding and Truncation: The tokenized inputs are padded to a fixed length and truncated if they exceed the maximum length, ensuring uniform input sizes for the model.
Model Architecture:

Embedding Layer: The input tokens are converted into dense vector representations using an embedding layer.
Transformer Encoder Layers: The embedding vectors are passed through multiple layers of self-attention and feed-forward networks. Each layer consists of:
Self-Attention Mechanism: This mechanism allows the model to focus on different parts of the input sequence, effectively capturing context and relationships between words.
Feed-Forward Networks: After the attention mechanism, the vectors are passed through fully connected feed-forward networks, applying non-linear transformations.
Distillation: While retaining the essential architecture of BERT, DistilBERT is trained to imitate BERT's output, leading to a smaller but powerful model.
Classification Head:

CLS Token Representation: For classification tasks, the output corresponding to the [CLS] token (the first token in the input sequence) is typically used as the aggregate representation of the input text.
Fully Connected Layer: The [CLS] token's output is fed into a fully connected layer, which maps the dense vector representation to the desired number of output classes (in this case, the different cyberbullying categories).
Softmax Activation: The final layer uses a softmax function to convert the logits into probabilities for each class, allowing the model to make a classification decision.
Loss Function:

Cross-Entropy Loss: During training, the model's predictions are compared to the true labels using cross-entropy loss, which measures the difference between the predicted probability distribution and the actual distribution.
Fine-Tuning DistilBERT
Fine-tuning involves training the pre-trained DistilBERT model on our specific dataset for the cyberbullying classification task. During fine-tuning:

Transfer Learning: The model leverages the knowledge gained during pretraining on large text corpora and adapts it to our smaller, domain-specific dataset.
Hyperparameter Tuning: Hyperparameters such as learning rate, batch size, and weight decay are optimized using Optuna, improving the model's performance.
Early Stopping: We implement early stopping to prevent overfitting, ensuring that the model does not train for more epochs than necessary if validation performance stops improving.
Summary of Model Architecture
In summary, DistilBERT for cyberbullying classification is a powerful and efficient model that combines the robustness of the BERT architecture with the benefits of a reduced model size. By fine-tuning this model on our specific dataset, we can effectively classify text into different categories of cyberbullying, leveraging the rich contextual understanding provided by the Transformer architecture.
