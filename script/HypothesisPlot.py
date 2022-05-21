# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import scipy.stats as scs
import numpy as np

from PlottingFunctions import PlottingFunctions
from ABTestingFunctions import ABTesting
PLTF = PlottingFunctions() 
ABT = ABTesting() 
#plot_null, plot_alt, show_area

class HypothesisPlot:
    def _init_(self):
        """
        Initializing HypothesisPlot class
        """
        
    def hypo_plot(self, Control, Exposed, bcr, mde, sig_level=0.05, show_power=False, show_beta=False,
                  show_alpha=False, show_p_value=False, show_legend=True):
        
        #create a plot object
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # define parameters to find pooled standard error
        X_C = bcr * Control
        X_E = (bcr + mde) * Exposed
        stderr = ABT.pooled_SE(Control, Exposed, X_C, X_E)
        
        # plot the distribution of the null and alternative hypothesis
        PLTF.plot_null(ax, stderr)
        PLTF.plot_alt(ax, stderr, mde)
        
        # set extent of plot area
        ax.set_xlim(-8 * stderr, 8 * stderr)
        
        # shade areas according to user input
        if show_power:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type='power')
        if show_alpha:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type='alpha')
        if show_beta:
            PLTF.show_area(ax, mde, stderr, sig_level, area_type='beta')
            
        # show p_value based on the binomial distributions for the two groups
        if show_p_value:
            null = ABT.ab_dist(stderr, 'control')
            p_value = ABT.p_val(Control, Exposed, bcr, bcr+mde)
            ax.text(3 * stderr, null.pdf(0),
                    'P-value={0:.3f}'.format(p_value),
                    fontsize=12, ha='left')
            
        # option to show legend
        if show_legend:
            plt.legend()

        plt.xlabel('d')
        plt.ylabel('PDF')
        plt.show()
        

    def abplot(self, N_A, N_B, bcr, d_hat, sig_level=0.05, show_power=False,
           show_alpha=False, show_beta=False, show_p_value=False,
           show_legend=True):
           
        """Example plot of AB test
        Example:
            abplot(n=4000, bcr=0.11, d_hat=0.03)
        Parameters:
            n (int): total sample size for both control and test groups (N_A + N_B)
            bcr (float): base conversion rate; conversion rate of control
            d_hat: difference in conversion rate between the control and test
                groups, sometimes referred to as **minimal detectable effect** when
                calculating minimum sample size or **lift** when discussing
                positive improvement desired from launching a change.
        Returns:
            None: the function plots an AB test as two distributions for
            visualization purposes
        """
        # create a plot object
        fig, ax = plt.subplots(figsize=(12, 6))

        # define parameters to find pooled standard error
        X_A = bcr * N_A
        X_B = (bcr + d_hat) * N_B
        stderr = ABT.pooled_SE(N_A, N_B, X_A, X_B)

        # plot the distribution of the null and alternative hypothesis
        PLTF.plot_null(ax, stderr)
        PLTF.plot_alt(ax, stderr, d_hat)

        # set extent of plot area
        ax.set_xlim(-8 * stderr, 8 * stderr)

        # shade areas according to user input
        if show_power:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type='power')
        if show_alpha:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type='alpha')
        if show_beta:
            PLTF.show_area(ax, d_hat, stderr, sig_level, area_type='beta')

        # show p_value based on the binomial distributions for the two groups
        if show_p_value:
            null = ABT.ab_dist(stderr, 'control')
            # p_value = ABT.p_val(N_A, N_B, bcr, bcr+d_hat)
            ax.text(3 * stderr, null.pdf(0),
                    'p-value={0:.3f}'.format(0.25917),
                    fontsize=12, ha='left')

        # option to show legend
        if show_legend:
            plt.legend()

        plt.xlabel('d')
        plt.ylabel('PDF')
        plt.show()

    def zplot(self, area=0.95, two_tailed=True, align_right=False):
        """Plots a z distribution with common annotations
        Example:
            zplot(area=0.95)
            zplot(area=0.80, two_tailed=False, align_right=True)
        Parameters:
            area (float): The area under the standard normal distribution curve.
            align (str): The area under the curve can be aligned to the center
                (default) or to the left.
        Returns:
            None: A plot of the normal distribution with annotations showing the
            area under the curve and the boundaries of the area.
        """
        # create plot object
        fig = plt.figure(figsize=(12, 6))
        ax = fig.subplots()
        # create normal distribution
        norm = scs.norm()
        # create data points to plot
        x = np.linspace(-5, 5, 1000)
        y = norm.pdf(x)

        ax.plot(x, y)

        # code to fill areas
        # for two-tailed tests
        if two_tailed:
            left = norm.ppf(0.5 - area / 2)
            right = norm.ppf(0.5 + area / 2)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

            ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                            where=(x > left) & (x < right))
            plt.xlabel('z')
            plt.ylabel('PDF')
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                    rotation=90, va="bottom", ha="right")
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                    fontsize=12, rotation=90, va="bottom", ha="left")
        # for one-tailed tests
        else:
            # align the area to the right
            if align_right:
                left = norm.ppf(1-area)
                ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
                ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                                where=x > left)
                plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                        fontsize=12, rotation=90, va="bottom", ha="right")
            # align the area to the left
            else:
                right = norm.ppf(area)
                ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
                ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                                where=x < right)
                plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                        fontsize=12, rotation=90, va="bottom", ha="left")

        # annotate the shaded area
        plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12,
                ha='center')
        # axis labels
        plt.xlabel('z')
        plt.ylabel('PDF')

        plt.show()   