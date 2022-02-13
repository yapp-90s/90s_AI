from django.test import TestCase

# import os
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gugongs.settings")

# from django.core.wsgi import get_wsgi_application
# application = get_wsgi_application()

# import django
# django.setup()

from views import *
import cv2

# Create your tests here.
def test():
    print("test start")

    img_dir = "/Users/a007/test/python/"

    image = cv2.imread(img_dir + "test_img.png", 1)  # 1 color, 2 grayscale, -1 alpha channel 포함


    cv2.imwrite(img_dir + "gamma.png", gammaImage(image, 1.5))
    cv2.imwrite(img_dir + "warm.png", warmImage(image))

    cv2.imwrite(img_dir + "cold.png", coldImage(image))    
    cv2.imwrite(img_dir + "guassian.png", guassian(image))
    
    print("test Done")


if __name__ == '__main__':
    test()