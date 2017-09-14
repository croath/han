Han
===

# Description

Han is a deep-learning project dealing with misspelled handwriting Chinese characters.

Its primary purpose is to find out the misspelled Chinese characters written by professional Chinese font designers, to review the result.

As there are about 5,000+ most common characters in Chinese, and regular Chinese font may contain more than 8,000. So there will be a chance for font designers, to miswrite it.

The project may not operate well on Chinese OCR purpose.

Tech details can be found with PDF: https://drive.google.com/file/d/0B-noE_nG9ncQV2R2M2F0eW5kbk0/view?usp=sharing

# How to use it

The project is writen with Python 3.6.

First of all, you should prepare some data to train. A good way to prepare train-data is to generate images from an existing font file.

`dump_ttf.sh` will do the most work.

Training progress will be triggered by `run.sh` script. And `rerun.sh` can help to run some new data on an existing checkpoint.

# Benchmark

With 20 different fonts and 8,877 images generated by those fonts as one epoch, all 15 epochs will be done within 7 hours on a K80 machine on Azure.

The accuracy will be higher than 99% on an entirely new font to detect wrong characters.

You can check the result with my model file: https://drive.google.com/file/d/0B-noE_nG9ncQY0ZZd29HWnNsUDQ/view?usp=sharing . It's a quantized model, can be loaded with `server.sh` script.

Feel free to send me an email if there's anything that confused you.
