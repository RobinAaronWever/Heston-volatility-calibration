# Option Pricing, Heston Model Calibration & Volatility Surface Fitting

> **Work in Progress:** This codebase is actively being refactored into modular Object-Oriented Python modules, with unit tests and benchmark suite integration in progress. Core numerical pricing engines and calibration algorithms are fully functional.

An end-to-end Python framework for empirical implied volatility surface construction, option pricing, and joint parameter calibration of the Heston stochastic volatility model across strike and maturity dimensions.

## Features
- **Data Pipeline:** Extracts and cleans real-time option chains across strikes and expiration dates via `yfinance`.
- **Implied Volatility Surface:** Constructs market implied volatility surfaces.
- **Pricing Engines:** Implements Black-Scholes and Heston option pricing models using numerical integration and the COS (Fourier cosine expansion) method.
- **Surface Calibration:** Minimizes root-mean-square error (RMSE) between market IVs and Heston model prices across the full strike-maturity grid using `scipy.optimize.differential_evolution`.


## Mathematical Framework
Under the Heston model, asset price $S_t$ and variance $v_t$ follow the coupled SDEs:

$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$
$$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v$$

where $d\langle W^S, W^v \rangle_t = \rho dt$.

Model parameters $\Theta = (\kappa, \theta, v_0, \xi, \rho)$ are calibrated by solving the non-linear least-squares optimization over $N$ market options:

$$\min_{\Theta} \sum_{i=1}^{N} w_i \left( C^{\text{market}}(K_i, T_i) - C^{\text{Heston}}(K_i, T_i; \Theta) \right)^2$$


## Discussion Points
Implied volatility is computed via the Newton-Raphson method:
$$\sigma_{i+1}=\sigma_{i}-\frac{BS(\sigma_{i})-V_{market}}{Vega(\sigma_{i})}$$

For some contracts, i.e., long maturities, deep ITM, and deep OTM, Vega tends to zero, causing the NR method not to converge. For this reason, we had to exclude a quarter of the dataset from the parameter calibration scheme. One fix would be to use the Brent method.

The optimal parameters obtained via the differential evolution optimization scheme had corner solutions; specifically, the long-term variance and the mean reversion rate obtained their maximal values, 0.25 and 1.5, respectively. 

The heat map of the percentage market price error shows that the optimization scheme predicts the prices well for a majority of the maturities and moneyness values, except for moneyness between 1.01 and 1.03. Furthermore, the concentration of errors seems to be around maturity times of 0.11 years or about 40 days. While parameter optimization achieved tight convergence in price space, about 1.3 % MSPE, transforming option prices back to Black-Scholes implied volatilities amplified errors to 16% MSPE, particularly driven by low-Vega OTM option contracts. Thus, simply minimizing the loss function dependent on option prices does not minimize the implied volatility. 



## Project Structure
├── Figures                 # Results obtained 
├── heston_calibration.py   # Main script for IV surface extraction, pricing, & calibration
└── README.md               # Project overview and documentation

