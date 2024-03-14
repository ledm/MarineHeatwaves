from PIL import Image, ImageFilter, ImageEnhance
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
    number_of_noisy_pixels = im.size[0]*im.size[1] * value/3. 

    for i in np.arange(int(number_of_noisy_pixels)):
        # from: https://onelinerhub.com/python-pillow/how-to-add-noise
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (0, 0, 0), # black
        )
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (255, 255, 255), # white
        )
        # grey:
        grey = int(random.random()*255)
        im.putpixel(
            (random.randint(0, im.size[0]-1), random.randint(0, im.size[1]-1)),
            (grey, grey, grey) # white        
        )        
    return im




def add_distortion_to_fig(temperature_anom, time, input_file='', output_file = ''):
    """
    Adds noise and sharp edges to an image.

    time is in decimal years.
    """
    #blend is high
    level = int(temperature_anom)
    
    if temperature_anom < 1.0:
        # no distortion for low heatwaves.        
        return input_file

    time = np.abs(time)
    if time > 0.5: 
        # too long since last new level.        
        return input_file

    # time_in_rad = time*180. # ie 0.5 years is 90 degrees
    # blend = math.cos(math.radians(time_in_rad))
    # np.clip(blend, 0., 1.) 

    time_in_rad = time * 2. * 90. # ie 0.5 years is 90 degrees
    # longer: 1 year is 90: 
    # time_in_rad = time * 90.
    # faster:
    # time_in_rad = time * 3* 90.
    time = np.clip(time, 0., 90.)

    # cos, so when level is 3 and t=0, blend is 100%.
    blend = np.clip(math.cos(math.radians(time_in_rad)), 0., 1.) # * level/3.
    np.clip(blend, 0., 1.) 
   
    #if input_file == output_file:
    #   print('Adding distortion in place to ', input_file, 'blend:', blend )#  'noise_level:', noise_level)

    im = Image.open(input_file)

    #filter = 
    #im1 = im.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # enhance sharpness, contrast and brightness in line with level.
    #im3 = ImageEnhance.Sharpness(i1)
    #im3 = im3.enhance(10.)
    im3 = ImageEnhance.Contrast(im)
    im3 = im3.enhance(1. + temperature_anom/15.)
    im3 = ImageEnhance.Brightness(im3)
    im3 = im3.enhance(1. + temperature_anom/15.)

    #im1 = add_noise(im1, noise_level)

    im1 = Image.blend(im, im3, blend)

    im1.save(output_file)

    return output_file

def test_distortion(temperature_anom, time, input_file='', output_file = ''):
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
    blend = math.cos(math.radians(time_in_rad)) * level/3.
    #blend = 
    np.clip(blend, 0., 1.) 
    #noise_level = (temperature_anom -0.75) * 0.01/3. 

    im = Image.open(input_file)

    for filter, fil_str in zip([
        ImageFilter.EDGE_ENHANCE_MORE,
        ImageFilter.SHARPEN,
        
        #ImageFilter.FIND_EDGES,
        #ImageFilter.EMBOSS,
        ],
        ['EDGE_ENHANCE_MORE',
         'SHARPEN',
         #'FIND_EDGES',
         #'EMBOSS'
         ]):
        #filter = ImageFilter.EDGE_ENHANCE_MORE

        im1 = im.filter(filter)
        im1 = Image.blend(im, im1, blend)
        #im1 = add_noise(im1, noise_level)
        
        output_file2 = output_file.replace('.png', '_'+fil_str+'.png')
        im1.save(output_file2)

def test_enhance(input_fn):
    im = Image.open(input_fn)
    # im3 = ImageEnhance.Sharpness(im)
    # im3 = im3.enhance(2.0)
    # output_file2 = input_fn.replace('.png', '_sharpness_2.png')
    # im3.save(output_file2)
   
    # im3 = ImageEnhance.Brightness(im)
    # im3= im3.enhance(2.0)
    # output_file2 = input_fn.replace('.png', '_brightness_2.png')
    # im3.save(output_file2)

    # im3 = ImageEnhance.Contrast(im)
    # im3 = im3.enhance(2.0)
    # output_file2 = input_fn.replace('.png', '_Contrast_2.png')
    # im3.save(output_file2)

    im3 = ImageEnhance.Sharpness(im)
    im3 = im3.enhance(5.0)
    im3 = ImageEnhance.Contrast(im3)
    im3 = im3.enhance(1.2)
    im3 = ImageEnhance.Brightness(im3)
    im3= im3.enhance(1.2)
    output_file2 = input_fn.replace('.png', '_all3x.png')
    im3.save(output_file2)   


    im3 = ImageEnhance.Sharpness(im)
    im3 = im3.enhance(30.0)
    #im3 = ImageEnhance.Contrast(im3)
    #im3 = im3.enhance(1.2)
    #im3 = ImageEnhance.Brightness(im3)
    #im3= im3.enhance(0.8)
    output_file2 = input_fn.replace('.png', '_all3x-0.8.png')
    im3.save(output_file2)   


if __name__ == "__main__":
    input_file = test_fn = "/users/modellers/ledm/workspace/MarineHeatwaves/images/test_distortion_2/daily_2045-08-10.png"

    test_enhance(input_file)
    #return
    #for time in [0., 0.4]: #np.arange(0., 0.7, 0.1):
    #  for level in [0,1,2,3]:
    #    output_file = test_fn.replace('.png', '_level'+str(level)+'_'+str(time)+'.png')
    #    test_distortion(level, time, input_file=input_file, output_file=output_file)
#
        # test_distortion(1, time, input_file=input_file, output_file=output_file)
        # test_distortion(2, time, input_file=input_file, output_file=output_file)
        # test_distortion(3, time, input_file=input_file, output_file=output_file)

