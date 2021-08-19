#!/usr/bin/env python3
# coding: UTF-8 
# Created by Cmoon

import pdfkit
import os


class Pdfmaker:
    def __init__(self):
        self.pdfpath = os.path.dirname(os.path.dirname(__file__)) + '/pdf'
        self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'

        # for filename in os.listdir(self.pdfpath):
        #     os.remove(self.pdfpath + '/' + filename)

    def write(self, text):
        with open(self.pdfpath + '/test.html', 'a') as f:
            f.write('<h2>' + text + '</h2>')
        self.make_pdf()

    def write_img(self):
        with open(self.pdfpath + '/test.html', 'a') as f:
            for filename in os.listdir(self.pdfpath):
                if '.jpg' in filename:
                    f.write('<img src="' + filename + '"><br>')
        self.make_pdf()

    def make_pdf(self):
        pdfkit.from_file(self.pdfpath + '/test.html', self.pdfpath + '/test.pdf')


if __name__ == '__main__':
    pdfmaker = Pdfmaker()
    pdfmaker.write('Testing writing cmd to pdf file.')
    pdfmaker.write_img()
    pdfmaker.make_pdf()
    pdfmaker.write('Testing writing again.')
    pdfmaker.make_pdf()
