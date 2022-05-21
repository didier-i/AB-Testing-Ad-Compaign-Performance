# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as scs
from HypothesisPlot import HypothesisPlot
HPP = HypothesisPlot() 
#hypo_plot

class DistributionPlots:
    def _init_(self):
        """
        Initializing DistributionPlots class
        """
        
    def cont_distribution(self, C_aware, C_total, C_cr, E_cr) -> None:
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.linspace(C_aware-49, C_aware+50, 100)
        y = scs.binom(C_total, C_cr).pmf(x)
        ax.bar(x, y, alpha=0.5)
        ax.axvline(x=E_cr * C_total, c='red', alpha=0.75, linestyle='--')
        plt.xlabel('Aware')
        plt.ylabel('Probability')
        plt.show()
        
    def cont_exp_distribution(self, C_aware, E_aware, C_total, E_total, C_cr, E_cr) -> None:
        fig, ax = plt.subplots(figsize=(12,6))
        xC = np.linspace(C_aware-49, C_aware+50, 100)
        yC = scs.binom(C_total, C_cr).pmf(xC)
        ax.bar(xC, yC, alpha=0.5)
        xE = np.linspace(E_aware-49, E_aware+50, 100)
        yE = scs.binom(E_total, E_cr).pmf(xE)
        ax.bar(xE, yE, alpha=0.5)
        plt.xlabel('Aware')
        plt.ylabel('Probability')
        #plt.show()
        
    def null_alt_distribution(self, C_total, E_total, C_cr, E_cr) -> None:
        bcr = C_cr
        mde = E_cr - C_cr
        HPP.hypo_plot(C_total, E_total, bcr, mde, show_power=True, show_beta=True, show_alpha=True, show_p_value=True)


    def null_alt_distribution1(self, SE_C, SE_E, cont_rate, exp_rate) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))        
        x = np.linspace(.35, .6, 1000)
        yC = scs.norm(cont_rate, SE_C).pdf(x)
        ax.plot(x, yC, label='Control')
        ax.axvline(x=cont_rate, c='red', alpha=0.5, linestyle='--')

        yE = scs.norm(exp_rate, SE_E).pdf(x)
        ax.plot(x, yE, label='Exposed')
        ax.axvline(x=exp_rate, c='blue', alpha=0.5, linestyle='--')

        plt.legend()
        plt.xlabel('Awareness Proportion')
        plt.ylabel('PDF')
        plt.show()