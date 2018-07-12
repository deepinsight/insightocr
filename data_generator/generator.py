#!/usr/bin/env python

import os
import random
import inspect
import glob
import string
import math
import sys
from pathlib import Path
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.join(str(Path.home()), "generated")
OUTPUT_NUM = 1000  # shall be multiple of OUTPUT_BATCH
OUTPUT_BATCH = 100
IMG_WIDTH = 200
IMG_HEIGHT = 32
MAX_LEFT_BLANK = 10
MIN_FONT_SIZE = 22
MAX_FONT_SIZE = IMG_HEIGHT - 4
MAX_TEXT_ANGLE = 5
TEXT_COLORS = [ImageColor.getrgb('black'), ImageColor.getrgb('blue')]

mainDir = None
backgrounds = None
fontsChinese = None
fontsEnglish = None
numChars = None
engChars = None
chiChars = None
commonChiChars = None
rareChiChars = None


'''
TODO:
- use similar font
- random space size between characters
- generate semantic text
'''


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))


def allowRotation(recWH, borderWH):
    '''
    Calculate the degree with which rotating a rectangle (size recWH) at center, the result is still contained by borderWH.
    '''
    assert(recWH[0] <= borderWH[0])
    assert(recWH[1] <= borderWH[1])
    L = math.hypot(recWH[0], recWH[1])
    W, H = borderWH[0], borderWH[1]
    a = math.atan(recWH[1]*1.0/recWH[0])
    bMax1 = math.pi/2-a
    if H < L:
        bMax1 = math.asin(H/L) - a
    bMax2 = math.pi/2
    if W < L:
        bMax2 = a - math.acos(W/L)
    bMax = min(bMax1, bMax2)
    bMax = max(bMax, 0)
    return math.degrees(bMax)


def getRoationSize(recWH, b):
    '''
    Caclulate the border after rotating recWH with specified counter-clockwise angle.
    '''
    # Calculation for angle==0 introduces precision errors.
    if b == 0:
        return recWH
    L = math.hypot(recWH[0], recWH[1])
    a = math.atan(recWH[1]*1.0/recWH[0])
    b = math.radians(b)
    W = L * max(abs(math.cos(b-a)), abs(math.cos(b+a)))
    H = L * max(abs(math.sin(b-a)), abs(math.sin(b+a)))
    # math.ceil returns the smallest integer which is greater than or equal to the input value.
    return (math.ceil(W), math.ceil(H))


def testAllowRotation():
    borderWH = (200, 32)
    recWHs = [(180, 32), (180, 20), (100, 20), (30, 20), (20, 20)]
    for recWH in recWHs:
        bMax = allowRotation(recWH, borderWH)
        borderWH2 = getRoationSize(recWH, bMax)
        print(f'{recWH}, {bMax}, {borderWH2}')


def putTextAndRotate(background, txt, fontName):
    '''
    Put text, rotate, and crop. Retures an img of size IMG_WIDTH * IMG_HEIGHT.
    '''
    # determine font size
    img = Image.open(background)
    fontSize = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
    font = ImageFont.truetype(fontName, fontSize)
    print(
        f'background: {os.path.basename(background)}, font: {os.path.basename(fontName)}, fontSize: {fontSize}')

    # determine the center of rectangular region IMG_WIDTH * IMG_HEIGHT.
    # This location will be used as text center too.
    width, height = img.size
    centX = random.randint(IMG_WIDTH/2, width - IMG_WIDTH/2)
    centY = random.randint(IMG_HEIGHT/2, height - IMG_HEIGHT/2)

    # determine the rotation angle
    txtWidth, txtHeight = font.getsize(txt)
    if txtWidth > IMG_WIDTH or txtHeight > IMG_HEIGHT:
        print(
            f'WARNING: text size ({txtWidth}, {txtHeight}) is too big', file=sys.stderr)
        return None
    bMax = allowRotation((txtWidth, txtHeight), (IMG_WIDTH, IMG_HEIGHT))
    bMax = min(MAX_TEXT_ANGLE, bMax)
    angle = (2 * random.random() - 1) * bMax
    borderW, borderH = getRoationSize((txtWidth, txtHeight), angle)
    assert(borderW <= IMG_WIDTH)
    assert(borderH <= IMG_HEIGHT)

    # put text
    txtX = centX - int(txtWidth/2)
    txtY = centY - int(txtHeight/2)
    draw = ImageDraw.Draw(img)
    if TEXT_COLORS is not None and len(TEXT_COLORS) != 0:
        txtColor = random.choice(TEXT_COLORS)
    else:
        txtColor = rndColor()
    draw.text((txtX, txtY), txt, fill=txtColor, font=font)

    # rotate image
    img = img.rotate(angle, resample=Image.BICUBIC,
                     expand=0, center=(centX, centY))

    # determine the crop region IMG_WIDTH * IMG_WIDTH which covers the text border.
    minCropX = max(0, centX+math.ceil(borderW/2)-IMG_WIDTH,
                   centX-math.ceil(borderW/2)-MAX_LEFT_BLANK)
    maxCropX = min(width-IMG_WIDTH, centX-math.ceil(borderW/2))
    minCropY = max(0, centY+math.ceil(borderH/2)-IMG_HEIGHT)
    maxCropY = min(height-IMG_HEIGHT, centY-math.ceil(borderH/2))
    if minCropX >= maxCropX:
        print(
            f'WARNING: cropX range ({minCropX}, {maxCropX}) is invalid', file=sys.stderr)
        return None
    cropX = random.randint(minCropX, maxCropX)
    cropY = random.randint(minCropY, maxCropY)
    img2 = img.crop((cropX, cropY, cropX+IMG_WIDTH, cropY+IMG_HEIGHT))
    return img2


def rndKind():
    '''
    numbers: 0.2,
    numbers + English characters: 0.2,
    numbers + English/Chinese characters: 0.6,

    rnd = random.random()
    if rnd < 0.2:
        return 0
    elif rnd < 0.4:
        return 1
    else:
        return 2
    '''
    return random.choices([0, 1, 2], cum_weights=[2, 4, 10])[0]


def generateText(kind, leng):
    text = ''
    font = None
    if kind == 0:
        font = random.choice(fontsEnglish)
        for i in range(leng):
            text += random.choice(numChars)
    elif kind == 1:
        font = random.choice(fontsEnglish)
        total_punc = 0
        for i in range(leng):
            # text += random.choice(engChars)
            while 1:
                ch = random.choice(engChars)
                if ch in string.punctuation:
                    if 3 * (total_punc+1) > leng or (len(text) != 0 and text[-1] in string.punctuation):
                        continue
                    total_punc += 1
                break
            text += ch
    else:
        font = random.choice(fontsChinese)
        for i in range(leng):
            # text += random.choice(chiChars)
            rnd = random.random()
            if rnd < 0.9:
                ch = random.choice(commonChiChars)
            else:
                ch = random.choice(rareChiChars)
            text += ch
    return font, text


def generate(total):
    global mainDir, fontsChinese, fontsEnglish, backgrounds, numChars, engChars, chiChars, commonChiChars, rareChiChars
    if mainDir is None:
        # http://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing
        selfPath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        mainDir, _ = os.path.split(selfPath)
    fontsChinese = glob.glob(os.path.join(mainDir, 'fonts_Chinese', '*.ttf')) + \
        glob.glob(os.path.join(mainDir, 'fonts_Chinese', '*.otf'))
    fontsEnglish = fontsChinese + glob.glob(os.path.join(
        mainDir, 'fonts_English', '*.ttf')) + glob.glob(os.path.join(mainDir, 'fonts_English', '*.otf'))
    backgrounds = glob.glob(os.path.join(mainDir, 'background', '*.png')) + \
        glob.glob(os.path.join(mainDir, 'background', '*.jpg'))
    numChars = string.digits + "."
    engChars = string.digits + string.ascii_letters + string.punctuation
    chiChars = engChars + "".join(filter(lambda c: c not in string.whitespace, open(
        os.path.join(mainDir, 'dict-chinese.txt')).read()))
    commonChiChars = chiChars[:len(engChars) + 3500]
    rareChiChars = chiChars[len(engChars) + 3500:]
    num = 0
    while num < total:
        kind = rndKind()
        # print(f'kind: {kind}')
        while 1:
            leng = random.randint(3, 20)
            font, text = generateText(kind, leng)
            background = random.choice(backgrounds)

            img2 = putTextAndRotate(background, text, font)
            if img2 is None:
                continue
            yield img2, text
            num += 1
            break


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = open(os.path.join(OUTPUT_DIR, "labels.txt"), 'w')
    labels.truncate()
    i = 0
    outDir = None
    for im, text in generate(OUTPUT_NUM):
        if i % OUTPUT_BATCH == 0:
            outDir = os.path.join(OUTPUT_DIR, f'{int(i/OUTPUT_BATCH)}')
            os.makedirs(outDir, exist_ok=True)
        outf = os.path.join(outDir, f'{i}.jpg')
        im.save(outf)
        labels.write(f'{i}.jpg\t{text}\n')
        print(f'done {i}.jpg, text: {text}')
        i += 1
    labels.close()


if __name__ == '__main__':
    main()
