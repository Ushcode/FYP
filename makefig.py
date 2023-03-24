 #!/usr/local/bin/python3

import svgutils.transform as sg
import sys

#create new SVG figure
fig = sg.SVGFigure("16cm", "6.5cm")

# load matpotlib-generated figures
fig1 = sg.fromfile('sigmoid_fit.svg')
fig2 = sg.fromfile('anscombe.svg')

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot2.moveto(280, 0, scale=0.5)

# add text labels
txt1 = sg.TextElement(25,20, "A", size=12, weight="bold")
txt2 = sg.TextElement(305,20, "B", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2])
fig.append([txt1, txt2])

# save generated SVG files
fig.save("fig_final.svg")

#method 2

from matplotlib import pyplot as plt
import subprocess, os

def plot_as_emf(figure, **kwargs):
    inkscape_path = kwargs.get('inkscape', "C://Program Files//Inkscape//inkscape.exe")
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')
        emf_filepath = os.path.join(path, filename+'.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
        os.remove(svg_filepath)

# Method  3

plt.savefig('whatever.eps', bbox_inches='tight')


# Annotate

ax.annotate('local max', xy=(3, 1),  xycoords='data',
xytext=(0.8, 0.95), textcoords='axes fraction',
arrowprops=dict(facecolor='black', shrink=0.05),
horizontalalignment='right', verticalalignment='top',
)
