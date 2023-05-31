import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Definindo o Tokenizer input data usando BERTimbau Base
tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", do_lower_case=False, model_max_length=512
)

loaded_model = TFAutoModelForSequenceClassification.from_pretrained("./sentiment")

# Caso queira testar outras sentenças, por favor não utilize caracteres especiais.
test_sentence = "otimo produto reconmendo a todods que quiserem comprar , entrega rapida pela americanas"


predict_input = tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")
tf_output = loaded_model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1)
labels = ["Negative", "Positive", "Neutral"]
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
print(labels[label[0]])
