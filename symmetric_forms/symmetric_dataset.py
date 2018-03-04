
import os
import numpy as np
import cairo
import math
from matplotlib import pyplot as plt

from sklearn.cross_validation import train_test_split


def load_data(width=28, height=28, test_size=0.20, debug=False):
    """ Generate symmetric image dataset with houses and boats
    """ 

    # Generate 16.000 different settings
    settings = []
    for obj in [0, 1]:
        for phi in [x / 10 for x in range(-20, 20, 1)]:
            for obj_width in [0.6, 0.5, 0.4, 0.3]:
                for obj_height in [0.2, 0.3]:
                    for x in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                        for y in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                            settings.append((obj, (x, y), phi, (obj_width, obj_height)))

    # Generate dataset
    x = []
    y = []
    for i in range(len(settings)):
        image, label = generate_image(width, height, settings[i])
        x.append(image)
        y.append(label)

        if debug:
            img_dbg = np.array(rgb_image).reshape(width, height, 3)
            plt.imshow(img_dbg, interpolation='nearest')
            plt.show()
    
    # Return train and test set
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return ((x_train, y_train), (x_test, y_test))


def generate_image(width, height, settings):
    """ Generate house or boat for given settings.

        :param width: Width of image
        :param height: Height of image
        :param settings: (object_id, (x, y), phi, (obj_width, obj_height))
    """
    obj_id, pos, phi, size = settings

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    ctx.scale(width, height)  # Normalizing the canvas
    ctx.set_source_rgb(0, 0, 0)

    # General configs (sharp lines, no alpha etc.)
    ctx.paint_with_alpha(0)
    ctx.set_antialias(True)

    # Black background
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    # Decide which object to paint
    if obj_id == 0:
        _paint_boat(ctx, math.pi / 2 * phi, pos, size)
    elif obj_id == 1:
        _paint_house(ctx, math.pi / 2 * phi, pos, size)
    else:
        raise NameError("Invalid object id %d." % obj_id)

    # Create rgb array out of abgr memoryview
    data = surface.get_data()
    abgr_image = np.array(data).reshape(-1, 4)
    rgb_image = abgr_image[:, 0:3][:,::-1]

    return rgb_image, obj_id


def _paint_house(ctx, phi, pos, size):
    """ Function to paint a house
    """
    x, y = pos
    width, height = size

    # Transform it into the middle, rotate it and at the end to the given position
    # calculated from the middle
    ctx.translate(0.5, 0.5)
    ctx.rotate(phi)
    ctx.translate(x - width / 2, y - height)

    # Triangle
    ctx.set_source_rgb(0.3, 1, 0.3)
    ctx.move_to(0.0, 0.0)
    ctx.line_to(0.0, height)
    ctx.line_to(width, height)
    ctx.close_path()
    ctx.fill()

    # Rectangle
    ctx.set_source_rgb(1, 0.3, 0.3)
    ctx.translate(0.0, height)
    ctx.rectangle(0.0, 0.0, width, height)
    ctx.fill()


def _paint_boat(ctx, phi, pos, size):
    """ Function to paint a boat
    """
    x, y = pos
    width, height = size
    triangle_height = 2.5*height

    # Transform it into the middle, rotate it and at the end to the given position
    # calculated from the middle
    ctx.translate(0.5, 0.5)
    ctx.rotate(phi)
    ctx.translate(x - width / 2, y - (height + triangle_height)/2)

    # Triangle
    ctx.set_source_rgb(0.3, 1, 0.3)
    ctx.move_to(0.0, 0.0)
    ctx.line_to(width/3.5, triangle_height)
    ctx.line_to(width-width/4.5, triangle_height-triangle_height/3)
    ctx.close_path()
    ctx.fill()

    # Rectangle
    ctx.set_source_rgb(1, 0.3, 0.3)
    ctx.translate(0.0, triangle_height)
    ctx.rectangle(0.0, 0.0, width, height)
    ctx.fill()