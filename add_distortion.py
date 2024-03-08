from PIL import Image, ImageFilter
import numpy as np
import random
import math

def add_salt_and_pepper(image, amount):
    """
    from https://stackoverflow.com/questions/59991178/creating-pixel-noise-with-pil-python
    """
    output = np.copy(np.array(image))
    print(output.shape)

    # add salt
    nb_salt = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
    print(len(coords[0]))
    output[coords] = [1]

    # add pepper
    nb_pepper = np.ceil(amount* output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
    output[coords] = 0

    return Image.fromarray(output)



def add_noise(im, value):
    if value == 0: 
        return im
    if value > 1.:
        assert 0
    number_of_noisy_pixels = im.size[0]*im.size[1] * value/2. 

    for i in np.arange(number_of_noisy_pixels):
        # from: https://onelinerhub.com/python-pillow/how-to-add-noise
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (0, 0, 0) # black
        )
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (255, 255, 255) # white
        )        
    return im




def add_distortion_to_fig(temperature_anom, time, input_file='', output_file = ''):
    """
    Adds noise and sharp edges to an image.

    time is in decimal years.
    """
    #blend is high
    level = int(temperature_anom)

    if level < 1.0:
        return # no distortion for low heatwaves.
    if time > 0.5: 
        return # too long since last new level.

    time_in_rad = time*180. # ie 0.5 years is 90 degrees
    blend = math.cos(math.radians(time_in_rad))
    np.clip(blend, 0., 1.)
    noise_level = level * 0.015/3.
    
    if input_file == output_file:
        print('Adding distortion in place to ', input_file, 'blend:', blend, 'noise_level:', noise_level)

    im = Image.open(input_file)

    if level in [1, 2]:
        filter = ImageFilter.EDGE_ENHANCE
    if level in [3, ]:
        filter = ImageFilter.EDGE_ENHANCE_MORE

    im1 = im.filter(filter)
    im1 = add_noise(im1, noise_level)
    im1 = Image.blend(im, im1, blend)
          
    im1.save(output_file)


if __name__ == "__main__":
    for time in np.arange(0., 0.7, 0.1):
        input_file = test_fn = "/users/modellers/ledm/workspace/MarineHeatwaves/images/test_distortion/daily_2045-08-10.png"
        output_file = test_fn.replace('.png', '_level'+str(level)+'_'+str(time)+'.png')
        test_distortion(0, time, input_file=input_file, output_file=output_file)
        test_distortion(1, time, input_file=input_file, output_file=output_file)
        test_distortion(2, time, input_file=input_file, output_file=output_file)
        test_distortion(3, time, input_file=input_file, output_file=output_file)

