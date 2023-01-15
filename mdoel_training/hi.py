from matplotlib import pyplot as plt
from reportlab.lib.units import cm
from reportlab.lib import pagesizes
from reportlab.platypus import Flowable, SimpleDocTemplate, Paragraph


class MatplotlibFlowable(Flowable):
    def __init__(self, figure):
        self.figure = figure

    def draw(self):
        self.figure.canvas.draw()
        self.canv.drawImage(self.figure.canvas.get_renderer()._renderer, 0, 0)

# create the pdf file
pdf_file = SimpleDocTemplate("results.pdf", pagesize=pagesizes.A4)

# create the plot
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')

# wrap the plot in a flowable
plt_flowable = MatplotlibFlowable(plt.gcf())

# add the text and the plot to the flowables
flowables = [Paragraph("This is some text"), plt_flowable]

# build the pdf file
pdf_file.build(flowables)
