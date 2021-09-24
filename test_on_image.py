from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

def style_image(image_path, model):

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(model))
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(Image.open(image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    fn = image_path.split("/")[-1]
    save_image(stylized_image, "images/outputs/stylized-" + fn)
