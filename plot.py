import numpy as np
from matplotlib import pyplot as plt

# cm to fraction (0-1)
def cmtf(cm, figsize_inch = 1):
    return cm/2.54/figsize_inch

# margins in cm
def add_axes_as_grid(fig, rows, cols, m_xl=0, m_yl=0, m_xr=0, m_yr=0, m_xc=0, m_yc=0):

    # margins
    m_xl = cmtf(m_xl, fig.get_figwidth())
    m_xc = cmtf(m_xc, fig.get_figwidth())
    m_xr = cmtf(m_xr, fig.get_figwidth())
    m_yl = cmtf(m_yl, fig.get_figheight())
    m_yc = cmtf(m_yc, fig.get_figheight())
    m_yr = cmtf(m_yr, fig.get_figheight())

    h = (1-m_yl-(m_yc*(rows-1))-m_yr)/rows
    w = (1-m_xl-(m_xc*(cols-1))-m_xr)/cols
    # left bot w h
    rect = [m_xl, m_yl, w, h]
    axes = []
    for i in np.arange(0, rows):
        axes.append([])
        rect[0] = m_xl
        rect[1] = m_yl+i*(h+m_yc)
        for j in np.arange(0, cols):
            axes[i].append(fig.add_axes(rect))
            rect[0] += w + m_xc

    return np.array(axes)

