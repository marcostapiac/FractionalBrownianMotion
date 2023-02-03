from utils.math_functions import snorm, slognorm, truncnorm, np, sinvgamma, gammaDist
from utils.plotting_functions import histogramplot, plt, plot
"""
lognormscale, lognorms = 0.8, .7
rvs = slognorm.rvs(scale=lognormscale, s=lognorms, size=2000)
axis = np.linspace(slognorm.ppf(0.001, scale=lognormscale, s=lognorms), slognorm.ppf(0.99, scale=lognormscale, s=lognorms), 2000)
pdfVals = slognorm.pdf(axis, scale=lognormscale, s=lognorms)
histogramplot(rvs=rvs, axis=axis, pdf_vals=pdfVals, xlabel="x", ylabel="pdf", plottitle="Log norm", plottlabel="")
plt.show()

truncnormMean, truncNormScale = 2.5, 2.
rvs = truncnorm.rvs(a=-truncnormMean/truncNormScale, b=np.inf, loc=truncnormMean, scale=truncNormScale, size=2000)
axis = np.linspace(truncnorm.ppf(0.01, a=-truncnormMean/truncNormScale, b=np.inf, loc=truncnormMean, scale=truncNormScale), truncnorm.ppf(0.99, a=-truncnormMean/truncNormScale, b=np.inf, loc=truncnormMean, scale=truncNormScale), 2000)
pdfVals = truncnorm.pdf(axis, a=-truncnormMean/truncNormScale, b=np.inf, loc=truncnormMean, scale=truncNormScale)
histogramplot(rvs=rvs, axis=axis, pdf_vals=pdfVals, xlabel="x", ylabel="pdf", plottitle="Trunc norm", plottlabel="")
plt.show()
"""
invGammaAlpha, invGammaBeta = 1., 1.
rvs = 1./gammaDist.rvs(a=invGammaAlpha, scale=invGammaBeta, size=2000)
axis = np.linspace(sinvgamma.ppf(0.0001, a=invGammaAlpha, scale=invGammaBeta), sinvgamma.ppf(0.8, a=invGammaAlpha, scale=invGammaBeta), 2000)
pdfVals = sinvgamma.pdf(axis, a=invGammaAlpha, scale=invGammaBeta)
print(max(rvs))
histogramplot(rvs=rvs, axis=axis, pdf_vals=pdfVals, xlabel="x", ylabel="pdf", plottitle="Inverse gamma", plottlabel="")
plt.show()