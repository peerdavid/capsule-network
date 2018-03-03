
import os
import numpy as np
import cairo
import math

from sklearn.cross_validation import train_test_split

def generate_images(debug=False):
    """ Generate symmetric image dataset
    """ 

    if debug and not os.path.exists("data"):
            os.makedirs("data")

    WIDTH, HEIGHT = 28, 28

    settings = []
    # Generate 16.000 different settings
    for isBoat in [True, False]:
        for phi in [x / 10 for x in range(-20, 20, 1)]:
            for width in [0.6, 0.5, 0.4, 0.3]:
                for height in [0.2, 0.3]:
                    for x in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                        for y in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                            settings.append((isBoat, phi, (width, height), (x, y)))

    x = []
    y = []
    for i in range(len(settings)):
        isBoat, phi, size, pos = settings[i]

        # ToDo: Random colors
        # ToDo: Take some images out for validation
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surface)

        ctx.scale(WIDTH, HEIGHT)  # Normalizing the canvas
        ctx.set_source_rgb(0, 0, 0)

        # General configs (sharp lines etc.)
        ctx.paint_with_alpha(0)
        ctx.set_antialias(True)

        # Black background
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        if isBoat:
            _paint_boat(ctx, math.pi / 2 * phi, pos, size)
        else:
            _paint_house(ctx, math.pi / 2 * phi, pos, size)

        # Create rgb array out of abgr memoryview
        data = surface.get_data()
        abgr_image = np.array(data).reshape(-1, 4)
        rgb_image = abgr_image[:, 0:3][:,::-1]

        x.append(rgb_image)
        y.append(0 if isBoat else 1)

        if debug:
            surface.write_to_png("data/%d.png" % i)
    
    # Return train and test set
    x, y = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return ((x_train, y_train), (x_test, y_test))


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