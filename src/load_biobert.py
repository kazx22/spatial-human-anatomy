from transformers import pipeline

MODEL_NAME = "Ishan0612/biobert-ner-disease-ncbi"

ner = pipeline(
    task="token-classification", model=MODEL_NAME, aggregation_strategy="simple"
)
print("NER pipeline loaded successfully")

text = "Patient has diabetes mellitus and hypertension."

results = ner(text)

print(results)
