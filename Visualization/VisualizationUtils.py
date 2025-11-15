from PIL import Image, ImageDraw, ImageFont
import os

def combine_images_with_corner_text(main_path, image_path=None, output_path=None, corner_texts=['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    # Get a list of image files in the given path
    if image_path is None:
        image_path = [f for f in os.listdir(main_path) if f.endswith('.jpg') or f.endswith('.png')]

    image_files = [os.path.join(main_path, f) for f in image_path if f.endswith('.jpg') or f.endswith('.png')]

    if output_path is None:
        output_path = main_path + 'combined_image.jpg'

    # Create a blank canvas to combine the images
    # Calculate the resolution of the combined image
    first_image = Image.open(image_files[0])
    resolution = (first_image.width * int(len(image_files)/2), first_image.height * int(len(image_files)/2))
    
    # Create the blank canvas with the calculated resolution
    combined_image = Image.new('RGB', resolution, color='white')


    # Set the font and size for the corner text
    font_path = None
    if os.name == 'nt':  # Windows
        font_path = 'arial.ttf'
    elif os.name == 'posix':  # Mac or Linux
        font_path = '/Library/Fonts/Arial.ttf'
    font = ImageFont.truetype(font_path, int(first_image.width/15))

    # Iterate over each image file
    for i, image_file in enumerate(image_files):
        # Open the image file
        image = Image.open(os.path.join(main_path, image_file))

        # # Resize the image to fit the canvas
        # image = image.resize((400, 300))

        # Calculate the position for the image on the canvas
        x = (i % 2) * first_image.width
        y = (i // 2) * first_image.height

        # Paste the image onto the canvas
        combined_image.paste(image, (x, y))


        # Calculate the position for the corner text
        text_x = x + int(first_image.width/10)
        text_y = y + int(first_image.height/10)
        # Get the corner text for the current image
        corner_text = corner_texts[i]
        # Create a draw object
        draw = ImageDraw.Draw(combined_image)
        
        # Draw the corner text on the image
        draw.text((text_x, text_y), corner_text, fill=(0, 0, 0), font=font)

    # Save the combined image with corner text
    combined_image.save(output_path)

# Define the path to the folder containing the images

main_path = '/Users/ayman/Library/CloudStorage/OneDrive-Personal/Notes/Papers/Photonics/E-coil/Results/'
output_path = '/Users/ayman/Library/CloudStorage/OneDrive-Personal/Notes/Papers/Photonics/E-coil/Results/combined_image'
os.makedirs(output_path, exist_ok=True)
images = [
['HeatMap_Defect thickness_vs_wavelength.png',
'defect_thickness_peaks_wavelegnth.png',
'defect_thickness_FullWidthHalfMax.png',
'defect_thickness_Quality_factor.png',
],
['HeatMap_Layer a thickness_vs_wavelength.png',
'layer_1_thickness_peaks_wavelegnth.png',
'layer_1_thickness_FullWidthHalfMax.png',
'layer_1_thickness_Quality_factor.png',
],
['HeatMap_Layer b thickness_vs_wavelength.png',
'layer_2_thickness_peaks_wavelegnth.png',
'layer_2_thickness_FullWidthHalfMax.png',
'layer_2_thickness_Quality_factor.png',
],
['HeatMap_N_vs_wavelength.png',
'N_peaks_wavelegnth.png',
'N_FullWidthHalfMax.png',
'N_Quality_factor.png',
],
['HeatMap_Layer a porosity_vs_wavelength.png',
'porosity_1_peaks_wavelegnth.png',
'porosity_1_FullWidthHalfMax.png',
'porosity_1_Quality_factor.png',
],
[
'HeatMap_Layer b porosity_vs_wavelength.png',
'porosity_2_peaks_wavelegnth.png',
'porosity_2_FullWidthHalfMax.png',
'porosity_2_Quality_factor.png',
],
['HeatMap_Inner core radius_vs_wavelength.png',
'rho0_peaks_wavelegnth.png',
'rho0_FullWidthHalfMax.png',
'rho0_Quality_factor.png',
],
['HeatMap_Volume fraction_vs_wavelength.png',
'Volume_fraction_peaks_wavelegnth.png',
'Volume_fraction_FullWidthHalfMax.png',
'Volume_fraction_Quality_factor.png',
'Volume_fraction_Sensitivity.png',
'Volume_fraction_Figure of Merit.png',
],
]
for image_list in images: 
    combine_images_with_corner_text(main_path, image_list, output_path + '/' 
                                    + image_list[1].split('_')[1]
                                    + image_list[1].split('_')[2]
                                    + '.jpg')
