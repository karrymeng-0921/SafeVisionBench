from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, StableDiffusionPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from models_IGMU import SDAModel


def generate_images(model_name, prompts_path, save_path, device='cuda:0',
                    guidance_scale=7.5, image_size=512, ddim_steps=100,
                    num_samples=10, from_case=0, till_case=1000000, base='1.4'):

    # Select base model
    if base == '1.4':
        model_version = "CompVis/stable-diffusion-v1-4"
    elif base == '1.5':
        model_version = "runwayml/stable-diffusion-v1-5"
    elif base == '2.0':
        model_version = "stabilityai/stable-diffusion-2"
    elif base == '2.1':
        model_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        model_version = "CompVis/stable-diffusion-v1-4"

    # Create base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_version,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)

    # Replace with modified model (SDAModel)
    if model_name == 'AdvUnlearn':
        concept = "nudity"
        unlearn_method = "AdvUnlearn"
        ckpt_path = "<path_to_advunlearn_checkpoint.pt>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'text_encoder'):
            pipe.text_encoder = model.text_encoder
        if hasattr(model, 'unet'):
            pipe.unet = model.unet
        if hasattr(model, 'vae'):
            pipe.vae = model.vae

        print(" AdvUnlearn model loaded successfully.")

    elif model_name == 'SPM':
        concept = "church"
        unlearn_method = "SPM"
        ckpt_path = "<path_to_spm_checkpoint.pt>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'unet'):
            pipe.unet = model.unet
        print(" SPM model loaded successfully.")

    elif model_name == 'ConceptPrune':
        concept = "Vincent van Gogh"
        unlearn_method = "ConceptPrune"
        ckpt_path = "<path_to_conceptprune_checkpoint.pt>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'unet'):
            pipe.unet = model.unet

        print(" ConceptPrune model loaded and neuron mask applied.")

    elif model_name == 'DoCoPreG':
        concept = "church"
        unlearn_method = "DoCoPreG"
        ckpt_path = "<path_to_docopreg_checkpoint.bin>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'unet'):
            pipe.unet = model.unet
        print(" DoCoPreG model loaded successfully.")

    elif model_name == 'MACE':
        concept = "church"
        unlearn_method = "MACE"
        ckpt_path = "<path_to_mace_checkpoint>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'text_encoder'):
            pipe.text_encoder = model.text_encoder
        if hasattr(model, 'unet'):
            pipe.unet = model.unet
        if hasattr(model, 'vae'):
            pipe.vae = model.vae

        print(" MACE model loaded successfully.")

    elif model_name == 'Receler':
        concept = "Vincent van Gogh"
        unlearn_method = "Receler"
        ckpt_path = "<path_to_receler_weights_folder>"

        model = SDAModel(unlearn_method=unlearn_method, concept=concept,
                         ckpt_path=ckpt_path, device=device)
        model.load_DM()

        if hasattr(model, 'unet'):
            pipe.unet = model.unet

        print(" Receler model loaded with Eraser module.")

    # Scheduler setup
    pipe.scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # Read prompts CSV
    df = pd.read_csv(prompts_path)
    os.makedirs(save_path, exist_ok=True)
    folder_path = os.path.join(save_path, f'{model_name.replace("diffusers-", "").replace(".pt", "")}')
    os.makedirs(folder_path, exist_ok=True)

    # Inference loop
    for _, row in df.iterrows():
        case_number = row.case_number
        if not (from_case <= case_number <= till_case):
            continue

        prompt = str(row.prompt)
        seed = int(row.evaluation_seed)

        generator = torch.Generator(device=device).manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_samples,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
            num_inference_steps=ddim_steps,
            generator=generator
        )

        images = output.images
        for num, im in enumerate(images):
            im.save(os.path.join(folder_path, f'{case_number}_{num}_sd{base}.png'))

    print(" All images generated successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers'
    )
    parser.add_argument('--model_name', help='name of the model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to CSV file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where images will be saved', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--base', type=str, default='1.4')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--till_case', type=int, default=1000000)
    parser.add_argument('--from_case', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--ddim_steps', type=int, default=100)
    args = parser.parse_args()

    generate_images(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        till_case=args.till_case,
        base=args.base
    )
