import torch
from model import load_model
from data_processing import extract_frames

def infer(video_path):
    # Load the trained model
    model = load_model()
    model.load_state_dict(torch.load("timesformer_asl.pth"))
    model.eval()

    # Perform inference
    with torch.no_grad():
        frames = extract_frames(video_path)
        outputs = model(frames.unsqueeze(0))
        predicted_label = outputs.logits.argmax(dim=1).item()
        print(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    infer("path_to_video")