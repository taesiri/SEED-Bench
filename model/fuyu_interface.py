# WIP 
import os
import numpy as np
from PIL import Image
from transformers import FuyuForCausalLM, AutoTokenizer
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor
import torch
from tqdm import tqdm

class MLM_Tester:
    def __init__(self, model_name='adept/fuyu-8b'):
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = FuyuForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=self.dtype).to(self.device)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)

    @staticmethod
    def resize_to_max(image, max_width=1080, max_height=1080):
        # Resize image to max dimensions
        width, height = image.size
        if width <= max_width and height <= max_height:
            return image

        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.LANCZOS)

    def process_prompt(self, question):
        # Format the prompt as needed
        return f"Question: {question}\nAnswer:"

    def predict(self, image, prompt):
        # Prepare image and prompt for prediction
        image = self.resize_to_max(image).convert("RGB")
        image_np = np.array(image)

        # Ensure the image is in HWC format
        if image_np.shape[-1] != 3:
            image_np = image_np.transpose((1, 2, 0))

        model_inputs = self.processor(text=prompt, images=[image_np])
        model_inputs = {
            k: v.to(dtype=self.dtype if torch.is_floating_point(v) else v.dtype, device=self.device)
            for k, v in model_inputs.items()
        }

        # Generate the prediction
        generation_output = self.model.generate(**model_inputs, max_new_tokens=2048)
        prompt_len = model_inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(generation_output[0][prompt_len:], skip_special_tokens=True)

 def forward(self, x):
        # Process the input data x and perform prediction using the Fuyu model
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        data_type = x['data_type']
        results = []

        if data_type == 'image':
            # Load and preprocess the image
            raw_image = Image.open(data_path).convert("RGB")
            processed_image = self.resize_to_max(raw_image)
            image_np = np.array(processed_image)

            # Ensure the image is in HWC format (Height, Width, Channels)
            if image_np.shape[-1] != 3:
                image_np = image_np.transpose((1, 2, 0))

            # Format the prompt
            prompt = self.process_prompt(question)
            model_inputs = self.processor(text=prompt, images=[image_np])

            # Convert inputs to the appropriate device and dtype
            model_inputs = {
                k: v.to(dtype=self.dtype if torch.is_floating_point(v) else v.dtype, device=self.device)
                for k, v in model_inputs.items()
            }

            # Generate predictions for each choice
            for choice in choices:
                input_ids = model_inputs['input_ids']
                # Append the choice to the input IDs
                choice_ids = self.tokenizer(choice, return_tensors='pt').input_ids.to(self.device)
                combined_input_ids = torch.cat((input_ids, choice_ids), dim=-1)

                # Generate the output
                generation_output = self.model.generate(input_ids=combined_input_ids, max_new_tokens=2048)
                prompt_len = input_ids.shape[-1]
                decoded_output = self.tokenizer.decode(generation_output[0][prompt_len:], skip_special_tokens=True)
                
                # Here we consider the output as a loss-equivalent, lower is better
                # Assuming the model output can be used to score the choices
                # (this part is hypothetical and would depend on how you want to evaluate the choices)
                results.append((choice, decoded_output))
        
        else:
            # Handle other data types if necessary
            pass
        
def build():
    return MLM_Tester()
