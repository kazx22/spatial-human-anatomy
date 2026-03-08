from transformers import pipeline

MODEL_NAME = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)

# sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease
# Ishan0612/biobert-ner-disease-ncbi -major issues with model. Cant label any entity. Did not work with pipeline. Issues with tokenization and entity recognition.

ner = pipeline(
    task="token-classification", model=MODEL_NAME, aggregation_strategy="simple"
)
print("NER pipeline loaded successfully")

# text = "The patient has hypertension and asthma and is taking metformin and aspirin."

text = """
The patient is a 65 year old male with a history of hypertension, type 2 diabetes, and asthma.
He reports chest discomfort and persistent headache for the past two days.
Current medications include metformin, aspirin, and atorvastatin.
Blood pressure remains elevated despite treatment with lisinopril.
"""

res_disease_raw = ner(text)

print("Input text:")
print(text)
print("\nDetected entities:")

res_disease = []
for entity in res_disease_raw:
    if entity["entity_group"] == "LABEL_1":
        res_disease.append(
            {
                "word": entity["word"],
                "entity_group": "DISEASE",
                "score": entity["score"],
                "start": entity["start"],
                "end": entity["end"],
            }
        )


# for entity in res_disease:
#     print(f"Text: {entity['word']}")
#     print(f"Label: {entity['entity_group']}")
#     print(f"Score: {entity['score']:.4f}")
#     print(f"Start: {entity['start']}")
#     print(f"End: {entity['end']}")
#     print("-" * 30)


MODEL_NAME = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"
ner = pipeline(
    task="ner", model=MODEL_NAME, tokenizer=MODEL_NAME, aggregation_strategy="simple"
)

res_chem_raw = ner(text)

res_chem = []
for entity in res_chem_raw:
    res_chem.append(
        {
            "word": entity["word"],
            "entity_group": "CHEMICAL",
            "score": entity["score"],
            "start": entity["start"],
            "end": entity["end"],
        }
    )


# print("Detected entities:")
# for entity in res_chem:
#     print(f"Text: {entity['word']}")
#     print(f"Label: {entity['entity_group']}")
#     print(f"Score: {entity['score']:.4f}")
#     print(f"Start: {entity['start']}")
#     print(f"End: {entity['end']}")
#     print("-" * 30)


result = res_disease + res_chem


for entity in result:
    print(f"Text: {entity['word']}")
    print(f"Label: {entity['entity_group']}")
    print(f"Score: {entity['score']:.4f}")
    print(f"Start: {entity['start']}")
    print(f"End: {entity['end']}")
    print("-" * 30)


# still having the issue of the model not recognizing entities. When there is more than one sentence, the model seems to struggle. It works better when there is only one sentence.
# I will need to look into this further and see if there are any solutions or workarounds.

# !--------Current Problems with the models--------!

# type as DISEASE

# This is a partial span error.
# The model should ideally detect type 2 diabetes, but instead only grabbed type.

# chest as DISEASE

# This is a false positive.
# It is over-predicting.

# lis, ##ino, ##pri, ##l

# This is a subtoken split problem.
# The chemical model found the medication, but did not merge the pieces into one clean span.

# So the models are usable, but they need post-processing.
