import torch
import torch.nn as nn
import clip


class CLIPZeroShot(nn.Module):
    def __init__(self, labels, device, model_name="ViT-B/32"):
        super().__init__()

        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        self.model = model
        self.preprocess = preprocess  
        self.device = device

        prompt_template="a photo of a {}"
        prompts = [prompt_template.format(c) for c in labels]
        text_tokens = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.register_buffer("text_features", text_features)

    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ self.text_features.T)
        return logits