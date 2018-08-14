#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Install dependencies:
# sudo pip install pathlib2 Pillow pygame scikit-image

import os
import re
import codecs
import shutil
import random
import inspect
import glob
import pygame
import pygame.locals
import pygame.image
import pygame.freetype
import pygame.transform
import pygame.draw
import pygame.surfarray
from pygame import Color
import numpy as np
from PIL import Image, ImageOps
try:
    # The goal of pathlib2 is to provide a backport of standard pathlib module.
    # refers to https://pypi.org/project/pathlib2/
    from pathlib2 import Path
except Exception:
    # pathlib is new in version 3.4.
    # refers to https://docs.python.org/3/library/pathlib.html
    from pathlib import Path

# https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
# The default encoding of Python 2 is 'ascii', Python 3 is 'utf-8'.
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

# http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_morphology.html
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage import io


OUTPUT_DIR = os.path.join(str(Path.home()), "generated")
OUTPUT_NUM = 1000000  # shall be multiple of OUTPUT_BATCH
OUTPUT_BATCH = 1000
IMG_WIDTH = 200
IMG_HEIGHT = 32
MAX_LEFT_BLANK = 10
MIN_FONT_SIZE = int(IMG_HEIGHT / 3)
MAX_FONT_SIZE = IMG_HEIGHT - 4
MAX_TEXT_ANGLE = 0
MIN_TEXT_LEN = 2
MAX_TEXT_LEN = 10

mainDir = None
fontsChinese = None
chiPhrases = None


def initChineseSource(fp):
    global chiPhrases
    patt = re.compile('[\s,;!?，。；！？]+')
    txt = codecs.open(fp, 'r', encoding='UTF-8',
                      errors='ignore').read()
    print(txt[:1024])
    chiPhrases = re.split(patt, txt)


def putTextAndRotate(txt, fontName):
    '''
    Put text, rotate, and crop. Retures an img of size IMG_WIDTH * IMG_HEIGHT.
    There shall be some small pixel gap (1~3) with left/top/bottom border.
    '''
    SURF_WIDTH = 800
    SURF_HEIGHT = 600
    surf = pygame.Surface((800, 600))
    surf.fill(Color(255, 255, 255, 0))  # white
    # determine font size
    fontSize = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
    font = pygame.freetype.Font(fontName, fontSize)
    if MAX_TEXT_ANGLE != 0:
        font.rotation = random.randint(-MAX_TEXT_ANGLE, MAX_TEXT_ANGLE)
    print(
        'font: %s, fontSize: %s, rotation: %s, text: %s' % (os.path.basename(fontName), fontSize, font.rotation, txt))

    surfRect = surf.get_rect()
    txtRect = font.get_rect(txt)
    while len(txt) > 0 and (txtRect.width >= IMG_WIDTH or txtRect.height >= IMG_HEIGHT):
        print('txtRect %s is not covered by %s*%s' %
              (txtRect, IMG_HEIGHT, IMG_WIDTH))
        txt = txt[:-1]
        txtRect = font.get_rect(txt)
    if len(txt) == 0:
        print('txt become empty')
        return None, None
    # txtX = random.randint(0, surfRect.width - IMG_WIDTH)
    # txtY = random.randint(0, surfRect.height - IMG_HEIGHT)
    txtX = SURF_WIDTH / 2
    txtY = SURF_HEIGHT / 2
    txtRect2 = font.render_to(surf, (txtX, txtY), txt)
    # assert(txtRect == txtRect2)
    '''
    pygame.draw.rect(surf, Color(255, 0, 0, 0), pygame.Rect(
        txtX, txtY, txtRect.width, txtRect.height), 1)
    '''
    '''
    constrains:
    0<= gapX <= 3
    cropX + gapX = txtX
    gapX >= -txtRect.left
    gapX + txtRect.width <= IMG_WIDTH

    0 <= gapY1 <= 3
    0 <= gapY2 <= 3
    gapY1 = txtY - cropY
    gapY1 + gapY2 + txtRect.height <= IMG_HEIGHT
    '''
    gapX = random.randint(max(0, -txtRect.left),
                          min(IMG_WIDTH-txtRect.width, 3))
    gapY1 = random.randint(0, min(3, int((IMG_HEIGHT - txtRect.height)/2)))
    gapY2 = random.randint(0, min(3, int((IMG_HEIGHT - txtRect.height)/2)))
    cropX = txtX - gapX
    cropY = txtY - gapY1
    ratioX = IMG_WIDTH*1.0/(txtRect.width + gapX)
    ratioY = IMG_HEIGHT*1.0/(txtRect.height + gapY1 + gapY2)
    assert(ratioX >= 1.0)
    assert(ratioY >= 1.0)
    ratio = min(ratioX, ratioY)
    cropWidth = int(IMG_WIDTH*1.0/ratio)
    cropHeight = int(IMG_HEIGHT*1.0/ratio)
    cropRect = (cropX, cropY, cropWidth, cropHeight)
    print(
        'txtRect: %s, gapX: %s, gapY1: %s, gapY2: %s, cropRect: %s, ratioX: %s, ratioY: %s' % (txtRect, gapX, gapY1, gapY2, cropRect, ratioX, ratioY))
    surf = surf.subsurface(cropRect)
    surf = pygame.transform.scale(surf, (IMG_WIDTH, IMG_HEIGHT))
    return surf, txt


# http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
# https://www.cnblogs.com/gongxijun/p/6114232.html
def noise_generator(noise_type, image):
    """
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        return image


def addNoiseAndGray(surf):
    # https://stackoverflow.com/questions/34673424/how-to-get-numpy-array-of-rgb-colors-from-pygame-surface
    imgdata = pygame.surfarray.array3d(surf)
    imgdata = imgdata.swapaxes(0, 1)
    # print('imgdata shape %s' % imgdata.shape)  # shall be IMG_HEIGHT * IMG_WIDTH
    imgdata2 = noise_generator('s&p', imgdata)

    img2 = Image.fromarray(np.uint8(imgdata2))
    # img2.save('/home/zhichyu/Downloads/2sp.jpg')
    grayscale2 = ImageOps.grayscale(img2)
    # grayscale2.save('/home/zhichyu/Downloads/2bw2.jpg')
    # return grayscale2

    array = np.asarray(np.uint8(grayscale2))
    # print('array.shape %s' % array.shape)
    selem = disk(random.randint(0, 1))
    eroded = erosion(array, selem)
    return eroded


def generateText(leng):
    text = ''
    font = random.choice(fontsChinese)
    while(1):
        phrase = random.choice(chiPhrases)
        if len(phrase) >= leng:
            maxStartPos = len(phrase)-leng
            allowedPos = list(filter(
                lambda pos: pos == 0 or ord(phrase[pos]) >= 256 or ord(phrase[pos-1]) >= 256, range(0, maxStartPos+1)))
            if len(allowedPos) == 0:
                continue
            i = random.choice(allowedPos)
            text = phrase[i:i+leng]
            break
    return font, text


def generate(total):
    global mainDir, fontsChinese
    assert(mainDir is not None)
    assert(fontsChinese is not None)
    num = 0
    while num < total:
        while 1:
            leng = random.randint(MIN_TEXT_LEN, MAX_TEXT_LEN)
            font, text = generateText(leng)
            surf, text = putTextAndRotate(text, font)
            if surf is None:
                continue
            img = addNoiseAndGray(surf)
            yield img, text
            num += 1
            break


def main():
    global mainDir, fontsChinese
    pygame.init()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)
    labels = open(os.path.join(OUTPUT_DIR, "labels.txt"), 'w')
    labels.truncate()
    i = 0
    chiIdx = 0
    outDir = None

    # http://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing
    selfPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
    mainDir, _ = os.path.split(selfPath)
    dirFonts = os.path.join(mainDir, 'fonts_Chinese')
    fnFonts = filter(lambda fn: os.path.splitext(fn)[1].lower() in [
                     '.ttf', '.otf'], os.listdir(dirFonts))
    fontsChinese = list(
        map(lambda fn: os.path.join(dirFonts, fn), fnFonts))

    chiFiles = sorted(glob.glob('newsgroup/corpus-*.txt'))
    outputPerChiFile = OUTPUT_NUM / len(chiFiles)
    initChineseSource(chiFiles[0])
    chiIdx += 1

    for im, text in generate(OUTPUT_NUM):
        if i % OUTPUT_BATCH == 0:
            outDir = os.path.join(OUTPUT_DIR, str(int(i/OUTPUT_BATCH)))
            os.makedirs(outDir)
        if i != 0 and i % outputPerChiFile == 0:
            initChineseSource(chiFiles[chiIdx])
            chiIdx += 1
        outf = os.path.join(outDir, '%s.jpg' % i)
        # pygame.image.save(im, outf) #pygame
        # im.save(outf) #PIL image
        io.imsave(outf, im)  # scikit-image
        labels.write('%s/%s.jpg\t%s\n' % (int(i/OUTPUT_BATCH),
                                          i, text))
        print('done %s.jpg, text: %s' % (i, text))
        i += 1
    labels.close()


if __name__ == '__main__':
    main()
