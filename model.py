import numpy as np
from peft import PeftModelForCausalLM
from transformers import GenerationConfig
from transformers import CLIPProcessor, CLIPModel
import torch


class ModelWithLoRA(PeftModelForCausalLM):
    def __init__(self, llama_model, peft_config, num_vector_tokens=64):
        super().__init__(llama_model, peft_config)
        self.num_vector_tokens = num_vector_tokens

        self.clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        self.clip_model.to(llama_model.device)

        self.llame_proj = torch.nn.Linear(
            512, self.config.hidden_size
        )

        self.llama_model = llama_model
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
    
    def generate_new_tokens(self, images, input_ids, max_new_tokens=30):
        concatenated_embeddings = torch.Tensor().to(self.model.device)
        for image in images:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.model.device)
            image_embeddings = self.clip_model.get_image_features(**inputs)
            projected_image_embeddings = self.llame_proj(image_embeddings.unsqueeze(0))

            concatenated_embeddings = torch.cat((concatenated_embeddings, projected_image_embeddings), dim=1)
        
        ids = torch.cat(concatenated_embeddings, self.llama_model.get_input_embeddings()(input_ids), dim=1)
        for i in range(max_new_tokens):
            outputs = super().forward(
                inputs_embeds=ids,
            )

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            y = torch.nn.functional.softmax(next_token_logits, dim=1)
            next_token = torch.argmax(y, dim=1)

            ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
    
        return ids

    def forward(self, images, input_ids, attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        b, s, c, h, w = images.size()

        images = images.view(b * s, c, h, w)

        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.model.device)
        image_embeddings = self.clip_model.get_image_features(**inputs)

        projected_image_embeddings = self.llame_proj(image_embeddings)

        projected_image_embeddings = projected_image_embeddings.view(b, s, -1)

        input_embeddings = self.llama_model.get_input_embeddings()(input_ids)

        concatenated_embeddings = torch.stack([torch.cat((projected_image_embeddings[i].unsqueeze(0), input_embeddings[i]), dim=1) for i in range(b)]).to(self.model.device)
        # concatenated_embeddings = torch.cat((projected_image_embeddings, input_embeddings), dim=1)
        concatenated_embeddings = concatenated_embeddings.view(b, -1, self.config.hidden_size)

        outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=concatenated_embeddings,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        loss = outputs.loss

        # logits = outputs.logits
        # next_token_logits = logits[:, -1, :]
        # y = torch.nn.functional.softmax(next_token_logits, dim=1)
        # next_token = torch.argmax(y, dim=1)

        return {"loss": loss}