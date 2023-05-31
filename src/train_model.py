import clear_data
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Limpar TensorFlow session // cache
tf.keras.backend.clear_session()

# Criar uma estratégia de distribuição para o tensorflow
strategy = tf.distribute.OneDeviceStrategy("GPU:0")  # Replace with the appropriate device

# Ler
df = pd.read_csv("src/clear_b2w.csv")
reviews = df["review_text_processed"].values.tolist()
labels = df["Sentiment"].tolist()

# Split the data into training and validation sets
training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
    reviews, labels, test_size=0.2
)

# Tokenize input data usando BERTimbau Base
tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", do_lower_case=False, model_max_length=512
)
train_encodings = tokenizer(training_sentences, truncation=True, padding=True)
val_encodings = tokenizer(validation_sentences, truncation=True, padding=True)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), training_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), validation_labels))

# Definir o modelo dentro do escopo da estratégia
with strategy.scope():
    # Load do modelo BERT pre-treinado
    model = TFAutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=3)

    # Definindo o otimizador e loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compilando o modelo
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    # Treinando o modelo
    # Para obter um resultado mais preciso, seria necessário aumentar o número de epochs e o batch size
    model.fit(
        train_dataset.shuffle(50).batch(4),
        epochs=2,
        batch_size=4,
        validation_data=val_dataset.shuffle(50).batch(4),
    )
    # Salvando o modelo treinado
    model.save_pretrained("./sentiment")
