# Multimodal Image/Video Captioning with Qwen2.5-VL
This project focuses on a multimodal caption generation project using the **Qwen2.5-VL-3B-Instruct** model.
It performs a **Zero-shot Inference** with the model, where the model generates natural language captions for the input image or video.

## Project Overview 
This repository demonstrates how to use a **Vision-Language Model (VLM)** to analyze visual content and produce descriptive captions.
The workflow includes: 
1) Loading the Qwen multimodal model 
2) Detecting whether the input file is an image or video 
3) Extracting frames from videos 
4) Processing visual inputs with the model 
5) Generating a caption 
6) Displaying the visual content with the generated description

## Model Configuration
The model is loaded with 4-bit quantization to reduce GPU memory usage.\
bnb_config = BitsAndBytesConfig( load_in_4bit=True,\
bnb_4bit_compute_dtype=torch.float16,\
bnb_4bit_quant_type="nf4",\
bnb_4bit_use_double_quant=True )

## Video Processing
For video inputs: 
1) The video is loded using Decord.
2) It is sampled into 8 frames.
3) The frames are converted to PIL images.
4) These PIL images are passed to the model for captioning.

## System Architecture 
1) Input file (image of video)
2) Detect file type
3) If image -> direct processing
4) If video -> extract frames
5) Process inputs using tokenizer and visual encoder
6) Pass to Qwen2.5-VL model
7) Generate caption
