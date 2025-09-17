from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# initialize the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image = Image.open("C:/Users/HP/Website&App for portfolio/AI-Playground/AI-Generated-Caption/images/senku.jpg")

# Prepare the image
inputs = processor(image, return_tensors="pt")
# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
 
print("Generated Caption:", caption)