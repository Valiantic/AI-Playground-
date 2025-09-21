import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = "C:/Users/HP/Website&App for portfolio/AI-Playground/AI-Generated-Caption/images/senku.jpg"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# Don't add question mark on the text 
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)

#tl:dr autoprocessor automatically loads the best up to date processor for the model you are using.
# better generation of image caption than blipprocessor