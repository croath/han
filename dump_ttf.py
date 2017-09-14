import argparse
from data_reader import create_label_list_from_file
from chn_converter import int_to_chinese
from subprocess import call
import os

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labellist', type=str, default='/Users/croath/Documents/labels.list', help='Labels list')
    parser.add_argument('--fontfile', type=str, default='/Users/croath/Documents/test_font.ttf', help='Location of font file (Type TTF)')
    parser.add_argument('--output', type=str, default='/Users/croath/Documents/output/', help='where to save the file')
    FLAGS, _ = parser.parse_known_args()

    label_list = create_label_list_from_file(FLAGS.labellist)

    for int_value in label_list:
        hex_value = hex(int_value)[2:]
        chn = int_to_chinese(int_value)

        filename = 'uni{}_{}.png'.format(str(hex_value), chn)
        call(['convert', '-fuzz', '5%', '-trim',
        '-background', 'white',
        '-fill', 'black',
        '-font', FLAGS.fontfile,
        '-pointsize', '100',
        'label:'+chn,
        os.path.join(FLAGS.output, filename)])
