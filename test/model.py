from transformers import TimesformerForVideoClassification

def load_model():
    # Load the TimeSFormer model from Hugging Face
    model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")
    return model