import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
import albumentations as albu
from albumentations.core.composition import Compose
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from networks.vit_seg_modeling_swin_encodecedge_FANet_UI import VisionTransformerSWIN as ViT_seg_SWIN_ENCODECEDGE
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import argparse
import logging

#FA-Net Part (Ashwini Added)
from utils_FANet import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data, load_test_data
    )
def otsu_mask(img, size):
        #img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = img.convert("L")
        img = np.array(img)
        img = cv2.resize(img, size)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = th.astype(np.int32)
        th = th/255.0
        th = th > 0.5
        th = th.astype(np.int32)
        return th

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./testinput/kvasir', help='root dir for validation volume data')  # for acdc volume_path=root_dir #Ashwini changed path
parser.add_argument('--dataset', type=str,
                    default='kvasir', help='experiment_name') #Ashwini Changed from Synapse
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network') #Ashwini Changed from 9
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=10000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num') #Ashwini changed from 3
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

cols1, cols2 = st.columns([1, 5])

with cols1:
    st.image('./nitp_logo.png')

with cols2:
    st.write("## National Institute of Technology Patna")

st.title("Automatic Polyp Segmentor")
st.sidebar.image("./polyp_images.png", width=300)
st.sidebar.write("### Automatic Polyp Segmentation System using MaST-UNet")
st.sidebar.write("\n**Created by:**  \nAshwini Kumar Upadhyay,  \nPhD Scholar, NIT Patna");
#st.sidebar.write("\n**Under the guidance of:**  \n Dr. Ashish Kumar Bhandari,  \nAssistant Professor, NIT Patna");
#st.sidebar.write("\n\n\n**Project By:**  \nShivam Singh(54)  \nDeependu Mishra(28)  \nSudhir Tiwari(59)  \nAshish Yadav(23)  \nNitesh Kumar(39)")

# def parse_args():
#     # Dummy function, no need for argparse in Streamlit
#     return {'image_path': None, 'model_name': 'MaS-TransUNet'}

def segment_image(image, pmask, model):
    # Load model configuration
    config = {
        'input_h': 256,
        'input_w': 256,
        'input_channels': 3,
        'num_classes': 1  # Assuming binary segmentation
    }

    # Move model to CPU
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    #previous mask
    #pmask = pmask

    # Apply transformations
    transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])
    transformed_image = transform(image=image_cv)['image']
    transformed_image = torch.unsqueeze(torch.from_numpy(transformed_image.transpose(2, 0, 1)), dim=0).float()

    # Move input image to CPU
    transformed_image = transformed_image.to(device)
    pmask = pmask.to(device)

    # Predict segmentation mask
    with torch.no_grad():
        output, lateral_edge, loss1, loss2 = model(transformed_image, pmask)
        output = torch.sigmoid(output).cpu().numpy()

    # Save segmented image
    output_image = output[0].transpose(1, 2, 0)
    output_image = (output_image * 255).astype('uint8')

    return output_image

def main():

    args = parser.parse_args()
    #device1 = torch.device('cpu')
    #print("Available architectures:", archs.__dict__)
    # Load the pre-trained model
    #model_name = args['model_name']
    model_name = 'MaS-TransUNet'
    #print("Available model architectures:", list(archs.__dict__.keys()))


    # if model_name not in archs.__dict__:
    #     raise ValueError(f"Model architecture '{model_name}' not found in the archs module.")
    
    num_classes = 1  # Assuming binary segmentation
    input_channels = 3  # Assuming RGB images

    # Load the pre-trained model with specified arguments
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    #model = archs.__dict__[model_name](num_classes=num_classes, input_channels=input_channels)
    model = ViT_seg_SWIN_ENCODECEDGE(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    #model.load_state_dict(torch.load('./model/model.pth'))
    model.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')))

    mri_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"], key=1)
    print('MRI_File location = ', mri_file)
    mask_file = st.file_uploader("Upload Mask Image", type=["png", "jpg", "jpeg"], key=2)
    print('Mask_File location = ', mask_file)

    #im, pmask = init_mask(mri_file, 256)
    
    size = (256,256)

    if mri_file is not None and mask_file is not None:
        mriImage = Image.open(mri_file)
        mriImage = mriImage.resize(size)
        mri_np = np.array(mriImage)

        maskImage = Image.open(mask_file)
        maskImage = maskImage.resize(size)

        col1, col2, col3 = st.columns(3)

        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        with col2:
            st.image(maskImage, caption='Uploaded Mask Image')
        
        pmask = otsu_mask(mriImage, size)
        pmask = rle_encode(pmask)

        pmask = " ".join(str(d) for d in pmask)
        pmask = str(pmask)

        pmask = rle_decode(pmask, size)

        pmask = np.expand_dims(pmask, axis=0)
        pmask = np.expand_dims(pmask, axis=0)
        pmask = pmask.astype(np.float32)
        pmask = np.transpose(pmask, (0, 1, 3, 2))
        #cv2.imwrite('./prev_masks/prev_mask.png', pmask) #Ashwini added for saving prev_masks
        pmask = torch.from_numpy(pmask)
        #pmask = pmask.to(device1)

        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            segmented_image = segment_image(mriImage, pmask, model)
            with col3:
                st.image(segmented_image, caption='Segmented Image')


    if mri_file is not None and mask_file is None:
        mriImage = Image.open(mri_file)
        mriImage = mriImage.resize(size)
        mri_np = np.array(mriImage)

        col1, col2 = st.columns(2)
        # st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        with col1:
            st.image(mriImage, caption='Uploaded MRI Image')

        pmask = otsu_mask(mriImage, size)
        pmask = rle_encode(pmask)
        pmask = " ".join(str(d) for d in pmask)
        pmask = str(pmask)
        pmask = rle_decode(pmask, size)
        pmask = np.expand_dims(pmask, axis=0)
        pmask = np.expand_dims(pmask, axis=0)
        pmask = pmask.astype(np.float32)
        pmask = np.transpose(pmask, (0, 1, 3, 2))
        #cv2.imwrite('./prev_masks/prev_mask.png', pmask) #Ashwini added for saving prev_masks
        pmask = torch.from_numpy(pmask)
        #pmask = pmask.to(device1)
        if st.button('Segment Image'):
            # segmented_image = segment_image(mri_np, "brain_UNet_woDS")
            # st.image(segmented_image, caption='Segmented Image', use_column_width=True)
            segmented_image = segment_image(mriImage, pmask, model)
            with col2:
                st.image(segmented_image, caption='Segmented Image')

if __name__ == '__main__':
    main()
