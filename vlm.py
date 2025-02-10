from peft import LoraConfig, get_peft_model
import torch
from transformers import CLIPImageProcessor, CLIPModel, LlamaForCausalLM, GenerationConfig


class BugaVLM(torch.nn.Module):
    def __init__(self, clip_model: CLIPModel, clip_processor: CLIPImageProcessor, projector: torch.nn.Module, 
                 llama_model: LlamaForCausalLM, path_mlp: torch.nn.Module, time_mlp: torch.nn.Module, num_vector_tokens: int=64) -> None:
        super().__init__()
        self.num_vector_tokens = num_vector_tokens

        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.clip_model.to(llama_model.device)

        self.llama_proj = projector
        self.llama_proj.to(llama_model.device)

        self.llama_model = llama_model
        self.llama_model.to(llama_model.device)

        self.flatten = torch.nn.Flatten()

        self.path_mlp = path_mlp
        self.path_mlp.to(llama_model.device)

        self.time_mlp = time_mlp
        self.time_mlp.to(llama_model.device)

        self.to(llama_model.device)
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            use_cache=False,
            do_sample=True,
            max_length=384,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            _from_model_config=False,
        )

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): The images.
        """
        b, s, c, h, w = images.size()

        images = images.view(b * s, c, h, w)

        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.llama_model.device)
        image_embeddings = self.clip_model.get_image_features(**inputs)

        projected_image_embeddings = self.llama_proj(image_embeddings)

        projected_image_embeddings = projected_image_embeddings.view(b, s, -1)

        for _ in range(20):
            outputs = self.llama_model.forward(
                inputs_embeds=projected_image_embeddings,
            )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            y = torch.nn.functional.softmax(next_token_logits, dim=1)
            next_token = torch.argmax(y, dim=1)

            next_token_embedding = self.llama_model.get_input_embeddings()(next_token)
            next_token_embedding = next_token_embedding.unsqueeze(1)

            projected_image_embeddings = torch.cat((projected_image_embeddings, next_token_embedding), dim=1)

        new_embeds = projected_image_embeddings[:, s:, :]
        flattened_embeddings = self.flatten(new_embeds)
        path_waypoints = self.path_mlp(flattened_embeddings)

        # I have in path_waypoints a tensor of 40x1 float, I to need to resize it to a 20x2 tensor so that i convert (x1, y1, x2, y2, ..., x20, y20) to [(x1, y1), (x2, y2), ..., (x20, y20)]
        path_waypoints = path_waypoints.view(b, 20, 2)

        for _ in range(10):
            outputs = self.llama_model.forward(
                inputs_embeds=projected_image_embeddings,
            )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            y = torch.nn.functional.softmax(next_token_logits, dim=1)
            next_token = torch.argmax(y, dim=1)

            next_token_embedding = self.llama_model.get_input_embeddings()(next_token)
            next_token_embedding = next_token_embedding.unsqueeze(1)

            projected_image_embeddings = torch.cat((projected_image_embeddings, next_token_embedding), dim=1)
        
        new_embeds = projected_image_embeddings[:, (s+20):, :]
        flattened_embeddings = self.flatten(new_embeds)
        time_waypoints = self.time_mlp(flattened_embeddings)

        time_waypoints = time_waypoints.view(b, 10, 2)

        predicted_waypoints = torch.cat((path_waypoints, time_waypoints), dim=1)

        return predicted_waypoints
    

def load_model() -> BugaVLM:
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.05
    )

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    llama_model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    projector = torch.nn.Linear(512, llama_model.config.hidden_size)
    path_mlp = torch.nn.Sequential(
        torch.nn.Linear(20 * llama_model.config.hidden_size, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 40)
    )
    time_mlp = torch.nn.Sequential(
        torch.nn.Linear(10 * llama_model.config.hidden_size, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 20)
    )

    # llama_model = get_peft_model(llama_model, lora_config)

    model = BugaVLM(clip_model, clip_processor, projector, llama_model, path_mlp, time_mlp)

    return model