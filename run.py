import sys
import torch
from PIL import Image

# Importing the required classes from the diffusers and clip_interrogator modules
from diffusers import StableDiffusionPipeline
from clip_interrogator import Config, Interrogator

if __name__ == "__main__":
    # Check if the number of command line arguments is correct
    if len(sys.argv) < 3:
        print("Usage: python run.py <first_path> <second_path>")
        sys.exit(1)

    # Define the length of prompts
    length_prompts = 3

    # Load the first image from the command line argument and convert it to RGB
    image_path = sys.argv[1]
    image = Image.open(image_path).convert('RGB')

    # Create an instance of the Interrogator class with the CLIP model ViT-L-14/openai and a caption maximum length of 16
    ci = Interrogator(
        Config(clip_model_name="ViT-L-14/openai", caption_max_length=16))

    # Generate a prompt for the content of the image using the Interrogator instance
    content_prompt = ci.interrogate(image)

    # Load the second image from the command line argument and convert it to RGB
    image_path = sys.argv[2]
    image = Image.open(image_path).convert('RGB')

    # Generate a prompt for the style of the image using the Interrogator instance
    style_prompt = ci.interrogate(image)

    # Free memory for further computations
    del ci
    torch.cuda.empty_cache()

    # Combine the content and style prompts to form the final prompt
    final_prompt = f"Image with {','.join(content_prompt.split(',')[:length_prompts])}, having style of {', '.join(style_prompt.split(',')[:length_prompts])}"

    """
    CompVis/stable-diffusion-v1-4
    runwayml/stable-diffusion-v1-5
    stabilityai/stable-diffusion-2-1-base
    stabilityai/stable-diffusion-2-1
    """
    # Load the StableDiffusionPipeline model from the "stabilityai/stable-diffusion-2-1" repository and set the torch data type to float16
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)

    # Set the pipeline to run on the GPU
    pipeline = pipeline.to("cuda")

    # Create a torch Generator for generating random noise
    generator = torch.Generator("cuda")

    # Use the StableDiffusionPipeline model to generate an image with the given prompt and random noise
    image = pipeline(final_prompt, num_inference_steps=50,
                     generator=generator).images[0]

    # Save the generated image to a file
    image.save('result.png')
