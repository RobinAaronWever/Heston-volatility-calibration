print("Starting Implied_Vol_T0.py...", flush=True)

#some cool packages
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import differential_evolution
import seaborn as sns

# Force pandas to display every column in terminal 
pd.set_option('display.max_columns', None)
# Expand the maximum width so columns don't wrap to a new line messily
pd.set_option('display.width', 1000)



def get_annualized_time(t, t_0, calendar_type="actual", include_market_close=True):
    # Convert t_0 to pd.Timestamp and strip timezone for uniform subtraction
    t_0_dt = pd.to_datetime(t_0)
    if hasattr(t_0_dt, "tz_convert") and t_0_dt.tzinfo is not None:
        t_0_dt = t_0_dt.tz_convert("UTC").tz_localize(None)

    # Convert t to pd.Timestamp or Series
    t_dt = pd.to_datetime(t)

    # Strip timezones if present
    if isinstance(t_dt, (pd.Series, pd.DatetimeIndex)):
        if t_dt.dt.tz is not None:
            t_dt = t_dt.dt.tz_convert("UTC").dt.tz_localize(None)
    elif hasattr(t_dt, "tzinfo") and t_dt.tzinfo is not None:
        t_dt = t_dt.tz_convert("UTC").tz_localize(None)

    # Handle Market Close adjustment (21:00 UTC)
    if include_market_close:
        if isinstance(t_dt, pd.Timestamp) and t_dt.hour == 0:
            t_dt = t_dt + pd.Timedelta(hours=21)
        elif isinstance(t_dt, pd.Series):
            # Apply +21h only to midnight entries in the Series
            midnight_mask = t_dt.dt.hour == 0
            t_dt = t_dt.copy()
            t_dt[midnight_mask] = t_dt[midnight_mask] + pd.Timedelta(hours=21)

    # Calculate time difference in fractional days
    if isinstance(t_dt, pd.Series):
        days_diff = (t_dt - t_0_dt).dt.total_seconds() / 86400.0
    else:
        days_diff = (t_dt - t_0_dt).total_seconds() / 86400.0

    # Normalize to annualized years
    if calendar_type == "trading":
        return days_diff / 252.0
    else:
        return days_diff / 365.25


#define function that plots a specific metric of our call or put option vs strike price
def V_K_T_plot(ticker_symbol, expiry_date, metric= "midPrice"):
    #ticker is the type of stock that we will look at, i.e ^SPX, AAPL, NVDA
    #expiry date is the maturity time of contract
    #metric is what we want to plot vs strike



    #fetch data
    ticker= yf.Ticker(ticker_symbol)
    expirations= ticker.options

    #Check if ticker has any expirations, if not return stops the code
    if not expirations:
        print(f"Error: No options data found for symbol '{ticker_symbol}'.")
        return 

    #Make sure expiry date is well defined
    if not expiry_date in expirations:
        old_expiry_date=expiry_date
        expiry_date=expirations[0]
        print(f"The expiry date, {old_expiry_date}, can not be used, we will use {expiry_date} instead")
        
    
    #Fetch option matrix 
    chain=ticker.option_chain(expiry_date)

    #define call and put dataframe
    calls_df=chain.calls.copy()
    puts_df=chain.puts.copy()

    #there are liquidity gaps, strike prices that have no bid or ask and thus price=0
    #We will artifically give them an interpolated price
    for df in [calls_df, puts_df]:
        # Select all numerical columns except strike
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
            "strike", errors="ignore"
        )

        for col in numeric_cols:
            #Turn 0s into NaNs
            df[col] = df[col].replace(0, np.nan)

            #Interpolate ONLY inside non-zero boundaries, to not change OTM 
            df[col] = df[col].interpolate(method="linear", limit_area="inside")

            #Restore deep OTM trailing NaNs back to 0.0
            df[col] = df[col].fillna(0.0)

    #define midPrice for analysis
    calls_df['midPrice']=(calls_df['bid']+calls_df['ask'])/2
    puts_df['midPrice'] =(puts_df['bid']+puts_df['ask'])/2


    #fetch underlying data for reference
    underlying_price = ticker.fast_info["lastPrice"]

    #check if metric is actually able to be used
    if metric not in calls_df.columns:
        print(
            f"Warning: Metric '{metric}' not found. Defaulting to 'midPrice'."
        )
        metric = "midPrice"

    #Now lets plot calls and puts 
    plt.figure(figsize=(11,6))

    plt.plot(calls_df['strike'],
             calls_df[metric],
             label="Calls",
            color="green",
            marker="o",
            linewidth=2,)
    plt.plot(puts_df['strike'], 
             puts_df[metric],
             label="Puts",
            color="red",
            marker="s",
            linewidth=2,)

    #plot underlying
    plt.axvline(
        x=underlying_price,
        color="black",
        linestyle="--",
        label=f"Current Stock Price (${underlying_price:.2f})",
    )

    #make it look pretty
    plt.title(
        f"{ticker_symbol.upper()} Options ({metric}) vs Strike Price\nExpiration Date: {expiry_date}"
    )
    plt.xlabel("Strike Price ($)")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block= True)


#V_K_T_plot(ticker_symbol="^SPX", expiry_date="2026-10-16", metric= "midPrice")


#Now define function that takes the implied volatility calculated by yfinance 
# only looking at the strikes you want
def get_yf_implied_vol(ticker_symbol, expiry_date, target_strikes):
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiry_date)
    
    # Reindex directly onto your Data_processing strikes
    calls_iv = chain.calls.set_index("strike")["impliedVolatility"].reindex(target_strikes)
    puts_iv = chain.puts.set_index("strike")["impliedVolatility"].reindex(target_strikes)
    
    # Interpolate missing values along the target grid
    calls_iv = calls_iv.interpolate(method="linear", limit_area="inside")
    puts_iv = puts_iv.interpolate(method="linear", limit_area="inside")
    
    return calls_iv.values, puts_iv.values



#We now want to define a function that will process the data
#gives back Call and Put price, Strike price, and expiry date used
def Data_processing(ticker_symbol, expiry_date, metric= "midPrice"):
    #ticker is the type of stock that we will look at, i.e ^SPX, AAPL, NVDA
    #expiry date is the maturity time of contract
    #metric is what we want to plot vs strike



    #fetch data
    ticker= yf.Ticker(ticker_symbol)
    expirations= ticker.options

    #Check if ticker has any expirations, if not return stops the code
    if not expirations:
        print(f"Error: No options data found for symbol '{ticker_symbol}'.")
        return None, None, None, None, None, None, None

    #Make sure there is dat with that specific expiry date
    if not expiry_date in expirations:
        old_expiry_date=expiry_date
        expiry_date=expirations[0]
        print(f"The expiry date, {old_expiry_date}, can not be used, we will use {expiry_date} instead")
        
    
    #Fetch option matrix 
    chain=ticker.option_chain(expiry_date)

    #define call and put dataframe
    calls_df=chain.calls.copy()
    puts_df=chain.puts.copy()

    #define midPrice for analysis
    calls_df['midPrice']=(calls_df['bid']+calls_df['ask'])/2
    puts_df['midPrice'] =(puts_df['bid']+puts_df['ask'])/2
    

    #there are liquidity gaps, strike prices that have no bid or ask and thus price=0
    #We will artifically give them an interpolated price
    for df in [calls_df, puts_df]:
        # Select all numerical columns except strike
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
            "strike", errors="ignore"
        )

        for col in numeric_cols:
            #Turn 0s into NaNs
            df[col] = df[col].replace(0, np.nan)

            #Interpolate ONLY inside non-zero boundaries, to not change OTM 
            df[col] = df[col].interpolate(method="linear", limit_area="inside")

            #Restore deep OTM trailing NaNs back to 0.0
            df[col] = df[col].fillna(0.0)


    #check if metric is actually able to be used
    if metric not in calls_df.columns:
        print(
            f"Warning: Metric '{metric}' not found. Defaulting to 'midPrice'."
            )
        metric = "midPrice"


    #The following part ensure the put and call have the same strike dataframe
    #
    calls_subset = calls_df[["strike", metric]].rename(
        columns={metric: "call_price"})
    puts_subset = puts_df[["strike", metric]].rename(
        columns={metric: "put_price"})

    # Merge on strike
    aligned_df = pd.merge(calls_subset, puts_subset, on="strike", how="outer")

    #Sort by strike to ensure ascending order
    aligned_df = aligned_df.sort_values("strike").reset_index(drop=True)

    # Interpolate missing values resulting from non-overlapping strikes
    aligned_df["call_price"] = (aligned_df["call_price"].interpolate(
        method="linear", limit_area="inside").fillna(0.0))
    aligned_df["put_price"] = (aligned_df["put_price"].interpolate(
        method="linear", limit_area="inside").fillna(0.0))

    # Extracted aligned series
    strikes = aligned_df["strike"]
    V_calls = aligned_df["call_price"]
    V_puts = aligned_df["put_price"]

    

    #We will also return the underlying history
    history= ticker.history(period= "1y")
    underlying_history=history[["Close"]].reset_index()
    underlying_history.columns = ["Date", "Value"]

    #time_stamp is when the data was taken, what day does the option prices refer to
    info= ticker.info
    regular_market_time=info.get("regularMarketTime")

    #check if the regular market time is defined 
    if regular_market_time:
        time_stamp = pd.to_datetime(regular_market_time, unit="s", utc=True)
    else:
        # Fallback: Use the exact current execution time if Yahoo's key is missing
        time_stamp = pd.Timestamp.now(tz="UTC")


    #fetch underlying data for reference
    underlying_price = ticker.fast_info["lastPrice"]
    

    return V_calls, V_puts, strikes, underlying_history, time_stamp, expiry_date, underlying_price

#Data_processing returns V_calls, V_puts, strikes, underlying_history, time_stamp, expiry_date, underlying_price

        




################################################################################################
#Black-Scholes, Implied Volatility
###########################################################################################





#Now we will start to analyse the data
#We first need to know the interest rate
def get_sp500_risk_free_rate():
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        if not hist.empty:
            quoted_yield = hist['Close'].iloc[-1]
            R = quoted_yield / 100.0
            return np.log(1.0 + R)
    except Exception:
        pass
    return 0.045  #Standard fallback rate (~4.5%)


#then the dividends yield
def get_dividend_yield(ticker_symbol):
    # SPX dividend yield is mirrored by SPY (~1.3% historical benchmark)
    if ticker_symbol.upper() in ["^SPX", "SPY"]:
        try:
            spy = yf.Ticker("SPY")
            div = spy.info.get("dividendYield", None)
            if div is not None and not np.isnan(div):
                return float(div)
        except Exception:
            pass
        return 0.013  # Baseline default 1.3%

    # Standard equity ticker extraction
    try:
        ticker = yf.Ticker(ticker_symbol)
        div = ticker.info.get("dividendYield", 0.0)
        return float(div) if div else 0.0
    except Exception:
        return 0.0

#get parameters 
r_sp500 = get_sp500_risk_free_rate()
q_sp500=0.0
#get_dividend_yield("^SPX")

#Calculate the historical volatility
def hist_vol(S_t, ts):
    dt=ts[1]-ts[0]
    y= np.std(np.diff(np.log(S_t)))/np.sqrt(dt)
    return y

#Standard Feynman-Kac Call option pricing:
def Call_price_FK(r, sigma, T, t_0, S_0, K, q=0.0):
    dt = T - t_0
    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)
    return S_0 * np.exp(-q * dt) * norm.cdf(d1) - K * np.exp(-r * dt) * norm.cdf(d2)

#Standard Feynman-Kac Put option pricing:
def Put_price_FK(r, sigma, T, t_0, S_0, K, q=0.0):
    dt = T - t_0
    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))
    d2 = d1 - sigma * np.sqrt(dt)
    return K * np.exp(-r * dt) * norm.cdf(-d2) - S_0 * np.exp(-q * dt) * norm.cdf(-d1)


#define Vega for call option for implied vol
def Vega(r, sigma, T, t_0, S_0, K, q=0.0):
    dt = T - t_0
    d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))
    return S_0 * np.exp(-q * dt) * np.sqrt(dt) * norm.pdf(d1)



def BS_NR_Vect(V, payoff, r, sig_imp_0, T, t_0, S_0, K, error, N_max, q=0.0):

    # Ensure V and K are 1D float arrays
    V = np.atleast_1d(np.asarray(V, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    
    # Ensure S_0 is a float scalar
    S_0 = float(S_0)

    # Initial guess array matching the dimension of V
    sig = np.atleast_1d(np.asarray(sig_imp_0, dtype=float)) * np.ones_like(V, dtype=float)

    # Active mask tracking non-converged options
    active = np.ones(V.shape, dtype=bool)
    N = 0

    while N < N_max and np.any(active):
        # Calculate pricing error g and Vega dg for active elements
        T_active = T if np.ndim(T) == 0 else np.asarray(T)[active]
        g = payoff(r, sig[active], T_active, t_0, S_0, K[active], q=q) - V[active]
        dg = Vega(r, sig[active], T_active, t_0, S_0, K[active], q=q)

        # Identify elements where Vega is too small to invert safely
        valid_vega = np.abs(dg) > 1e-8
        
        # If Vega is too small, stop iterating on those specific options (set NaNs)
        if not np.all(valid_vega):
            # Extract active indices where vega is invalid
            active_indices = np.where(active)[0]
            invalid_indices = active_indices[~valid_vega]
            sig[invalid_indices] = np.nan
            active[invalid_indices] = False
            
            # Re-evaluate remaining active elements
            if not np.any(active):
                break
            g = g[valid_vega]
            dg = dg[valid_vega]

        # Calculate Newton step and cap maximum step size
        step = g / dg
        step = np.clip(step, -0.5, 0.5)  # Prevents jumping by more than 50% vol per step

        # Apply Newton step
        sig[active] -= step

        # Safeguard: Clip volatility into realistic financial bounds [0.1%, 500%]
        sig[active] = np.clip(sig[active], 1e-3, 5.0)

        # Update active mask where step size exceeds tolerance
        active[active] = np.abs(step) > error
        N += 1

    return sig
#BS_NR_Vect returns sig

#define objective for Brents method
def objective(sigma, V_market, payoff, r, T, t_0, S_0, K, q=0.0):
    return payoff(r, sigma, T, t_0, S_0, K, q=q) - V_market


#define function that calculates the option price from hist_vol and plot it
def plot_option_price_from_hist_vol(ticker_symbol, expiry_date, metric= "midPrice"):
#do data processing
    (
    V_calls,
    V_puts,
    strikes,
    underlying_history,
    time_stamp,
    expiry_date,
    underlying_price
    ) = Data_processing("^SPX", expiry_date, "midPrice")

    #calc historical vol
    vol_0=hist_vol(underlying_history['Value'], get_annualized_time(underlying_history['Date'], time_stamp))

    #calc BS
    V_calls_BS=Call_price_FK(r_sp500, vol_0, get_annualized_time(expiry_date, time_stamp), 0, underlying_price, strikes, q_sp500)
    V_puts_BS=Put_price_FK(r_sp500, vol_0, get_annualized_time(expiry_date, time_stamp), 0, underlying_price, strikes, q_sp500)


    #Now plot
    plt.plot(strikes,
             V_calls_BS,
             label="BS Calls",
             color="green",
             #marker="o",
             linewidth=2,)
    plt.plot(strikes, 
             V_puts_BS,
             label="BS Puts",
             color="red",
             #marker="s",
             linewidth=2,)
    plt.plot(strikes, 
             V_calls,
             label='Dataset Calls', 
             color= '#d62728')
    plt.plot(strikes, 
             V_puts, 
             label= 'Dataset Puts',
             color= '#1f77b4')

    #plot underlying
    plt.axvline(
        x=underlying_price,
        color="black",
        linestyle="--",
        label=f"Current Stock Price (${underlying_price:.2f})",
        )


    plt.title(
            f"{ticker_symbol.upper()} Options Price vs Strike Price\nExpiration Date: {expiry_date} calculated from historical vol"
        )
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Option Price ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block= True)


#Now lets define the function that takes the input and calculates the implied volatility
def implied_vol_NR(ticker_symbol, expiry_date, metric="midPrice", 
                   r=r_sp500, error=10**(-4), N_max=10**5):
    # Fetch data
    ( 
        V_calls,
        V_puts,
        strikes, 
        underlying_history,
        time_stamp,
        actual_expiry_date, 
        underlying_price,
    ) = Data_processing(ticker_symbol, expiry_date, metric)

    # Safety check if Data_processing returns None
    if V_calls is None:
        return None, None, None, None, None, None, None, None

    # give initial guess of volatility
    vol_0 = 0.2
    # Correctly format T, t_0 using actual returned expiration date
    T = get_annualized_time(actual_expiry_date, time_stamp)
    t_0 = 0

    # Calculate implied volatility
    imp_vol_calls = BS_NR_Vect(V_calls, Call_price_FK, r, vol_0, T, t_0, underlying_price, strikes, error, N_max, q_sp500)
    imp_vol_puts = BS_NR_Vect(V_puts, Put_price_FK, r, vol_0, T, t_0, underlying_price, strikes, error, N_max, q_sp500)

    # Drop NaNs
    valid_calls = ~np.isnan(imp_vol_calls)
    valid_puts = ~np.isnan(imp_vol_puts)

    return imp_vol_calls, imp_vol_puts, V_calls, V_puts, strikes, valid_calls, valid_puts, T


#implied_vol_NR returns imp_vol_calls, imp_vol_puts, strikes, valid_calls, valid_puts, T

def plot_implied_vol_NR(ticker_symbol, expiry_date, metric="midPrice", 
                        r=r_sp500, error=10**(-4), N_max=10**5):
    
    # Updated: Now unpacks 6 variables (added T_exp at the end)
    imp_vol_calls, imp_vol_puts, V_calls, V_puts, strikes, valid_calls, valid_puts, T_exp = implied_vol_NR(
        ticker_symbol, expiry_date, metric=metric, r=r, error=error, N_max=N_max
    )

    # Re-fetch metadata for the plot labels/underlying price
    _, _, _, _, time_stamp, actual_expiry_date, underlying_price = Data_processing(ticker_symbol, expiry_date, metric)

    # Now lets compare this to the ones from yfinance
    yfin_iv_calls, yfin_iv_puts = get_yf_implied_vol(ticker_symbol, actual_expiry_date, strikes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(strikes[valid_calls], imp_vol_calls[valid_calls], label="Calls")
    plt.plot(strikes[valid_puts], imp_vol_puts[valid_puts], label="Puts")
    plt.plot(strikes[valid_calls], yfin_iv_calls[valid_calls], label="Calls yfin")
    plt.plot(strikes[valid_puts], yfin_iv_puts[valid_puts], label="Puts yfin")

    # Plot underlying
    plt.axvline(
        x=underlying_price,
        color="black",
        linestyle="--",
        label=f"Current Stock Price (${underlying_price:.2f})",
    )

    plt.title(f"Implied Volatility vs Strike Price \nExpiration Date: {actual_expiry_date}")
    plt.xlabel("Strike Price ($)")
    plt.ylabel("Implied Volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=True)


#Now we can finally calculate imp vol surface

def imp_vol_surface_NR(
    ticker_symbol,
    metric="midPrice",
    r=r_sp500,
    error=1e-4,
    N_max=1000,
    plot=False,
    max_expirations=None,
):
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options

    if not expirations:
        print(f"Error: No expiration dates found for symbol '{ticker_symbol}'.")
        return pd.DataFrame(), pd.DataFrame()

    if max_expirations is not None:
        expirations = expirations[:max_expirations]

    calls_data = []
    puts_data = []

    for exp_date in expirations:
        try:
            # Fetch IV and T directly from single call
            res = implied_vol_NR(ticker_symbol, exp_date, metric=metric, r=r, error=error, N_max=N_max)
            if res[0] is None:
                continue

            iv_c, iv_p, V_calls, V_puts, strikes, valid_c, valid_p, T = res

            # Skip near-zero or expired maturities
            if T <= 0.005:
                continue

            # Append valid Call data points
            for k, iv, price in zip(strikes[valid_c], iv_c[valid_c], V_calls[valid_c]):
                if 0.01 < iv < 3.0 and price> 0.0:
                    calls_data.append({'Strike': k, 'T': T, 'IV': iv, 'Market_Price': price})

            # Append valid Put data points
            for k, iv, price in zip(strikes[valid_p], iv_p[valid_p], V_puts[valid_p]):
                if 0.01 < iv < 3.0 and price > 0.0:
                    puts_data.append({'Strike': k, 'T': T, 'IV': iv, 'Market_Price': price})

        except Exception as e:
            # Print error so you know if/why a specific date fails
            print(f"Skipped {exp_date} due to error: {e}")
            continue

    df_calls = pd.DataFrame(calls_data)
    df_puts = pd.DataFrame(puts_data)

    if df_calls.empty or df_puts.empty:
        print("No valid option data extracted to render surface.")
        return df_calls, df_puts

    def render_3d_surface(df, title_type):
        grid_strike = np.linspace(df['Strike'].min(), df['Strike'].max(), 50)
        grid_T = np.linspace(df['T'].min(), df['T'].max(), 50)
        K_mesh, T_mesh = np.meshgrid(grid_strike, grid_T)

        IV_mesh = griddata(
            (df['Strike'], df['T']), 
            df['IV'], 
            (K_mesh, T_mesh), 
            method='linear'
        )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            K_mesh, T_mesh, IV_mesh, 
            cmap='viridis', edgecolor='none', alpha=0.85
        )

        ax.set_xlabel('Strike Price ($)', labelpad=10)
        ax.set_ylabel('Time to Expiration (Years)', labelpad=10)
        ax.set_zlabel('Implied Volatility', labelpad=10)
        ax.set_title(f"{ticker_symbol.upper()} {title_type} Implied Volatility Surface", fontsize=14)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Volatility')
        plt.show()

    if plot:
        render_3d_surface(df_calls, "Call")
        render_3d_surface(df_puts, "Put")

    return df_calls, df_puts

#imp_vol_surface_NR returns df_calls, df_puts


#################################
#Test for Implied Volatility
##############################
#plot_option_price_from_hist_vol("^SPX", "2026-10-16", "midPrice")

#parameter estimations
#define the variables we want to look at
print("Loading initial option data...", flush=True)
(
    V_calls,
    V_puts,
    strikes,
    underlying_history,
    time_stamp,
    expiry_date,
    underlying_price
) = Data_processing("^SPX", "2026-10-16", "midPrice")
print("Initial option data loaded.", flush=True)



#vol_0=hist_vol(underlying_history['Value'], get_annualized_time(underlying_history['Date'], time_stamp))
#(imp_vol_calls, imp_vol_puts, V_calls, V_puts, strikes, valid_calls, valid_puts, T)=implied_vol_NR( "^SPX", "2026-10-16", "midPrice", r=r_sp500, error=10**(-4), N_max=10**5)
#print(imp_vol_calls)
#plot_implied_vol_NR( "^SPX", "2026-10-16", "midPrice", r=r_sp500, error=10**(-4), N_max=10**5)

print("Building implied-volatility surface...", flush=True)
imp_vol_surface_calls, imp_vol_surface_puts = imp_vol_surface_NR(
    "^SPX",
    metric="midPrice",
    r=r_sp500,
    error=1e-4,
    N_max=1000,
    plot=True,
    max_expirations=5,
)
print("Implied-volatility surface built.", flush=True)

print(imp_vol_surface_calls.head(), flush=True)

    



################################################################################################
#Heston Model, Parameter Fitting
###########################################################################################

#Now we develop the cos method applied to the Heston model
#g is now differently defined to the one in book to combat ill defined sqrt of complex number
#Furthermore, we use sigma_std= np.sqrt(v_bar* tau)

def COS_method_Heston(r, T, t_0, S_0, K, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=10, N_k=160):
    """
    Prices European options under the Heston model using the COS method.
    Domain truncation [a, b] is calculated via sigma_std = sqrt(v_bar * tau).
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    tau = np.atleast_1d(np.asarray(T, dtype=float) - t_0)
    tau, K = np.broadcast_arrays(tau, K)
    
    #Truncation domain [a, b] centered around X_0 = ln(S_0 / K)
    X_0 = np.log(S_0 / K)
    sigma_std = np.sqrt(v_bar * tau)
    
    a = X_0 - L * sigma_std
    b = X_0 + L * sigma_std
    
    # Fourier frequencies
    k = np.arange(N_k)
    k_grid = k[:, None]
    a_grid = a[None, :]
    b_grid = b[None, :]
    u_k = k_grid * np.pi / (b_grid - a_grid)
    
    # Heston Characteristic Function
    b_term = kappa - 1j * rho_xv * gamma * u_k
    D1 = np.sqrt(b_term**2 + (u_k**2 + 1j * u_k) * gamma**2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = (b_term - D1) / (b_term + D1)
    # The zero frequency has phi(0)=1; the direct expression is 0/0 there.
    g[0] = 0.0
    
    exp_D1 = np.exp(-D1 * tau)
    D_minus = b_term - D1
    
    fact1 = np.exp(
        1j * u_k * r * tau + 
        (v_0 / gamma**2) * ((1 - exp_D1) / (1 - g * exp_D1)) * D_minus
    )
    fact2 = np.exp(
        (kappa * v_bar / gamma**2) * (
            tau * D_minus - 2 * np.log((1 - g * exp_D1) / (1 - g))
        )
    )
    phi_k = fact1 * fact2
    
    # Payoff Integrals (chi and psi)
    def chi(c, d):
        return (
            np.cos(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - np.cos(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
            + u_k * np.sin(k_grid * np.pi * (d - a_grid) / (b_grid - a_grid)) * np.exp(d)
            - u_k * np.sin(k_grid * np.pi * (c - a_grid) / (b_grid - a_grid)) * np.exp(c)
        ) / (1 + u_k**2)

    def psi(c, d):
        k_safe = np.where(k_grid == 0, 1.0, k_grid)
        val_nonzero = ((b - a) / (k_safe * np.pi)) * (
            np.sin(k_safe * np.pi * (d - a_grid) / (b_grid - a_grid)) - 
            np.sin(k_safe * np.pi * (c - a_grid) / (b_grid - a_grid))
        )
        return np.where(k_grid == 0, d - c, val_nonzero)

    # Normalized Payoff Coefficients U_k
    if option_type.lower() == 'call':
        U_k = (2 / (b - a)) * (chi(0, b) - psi(0, b))
    elif option_type.lower() == 'put':
        U_k = (2 / (b - a)) * (-chi(a, 0) + psi(a, 0))
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
        
    # Summation
    exp_k = np.exp(1j * k_grid * np.pi * (X_0[None, :] - a_grid) / (b_grid - a_grid))
    term = phi_k * U_k * exp_k
    sum_val = np.sum(term, axis=0) - 0.5 * term[0]
    
    return K * np.exp(-r * tau) * sum_val.real



#Now we apply optimization schemes to fit the data
# We will first do global schemes and perhaps later local schemes when our initial guess is good

#first we will fix v_bar=sigma^{2}_ ATM, 
# the square of the implied volatility of the longest maturity date
#And next the kappa=2 for standard SnP500
kappa_fixed=2

def extract_v_bar(imp_vol_surface, S_0, r=r_sp500):
    #Extracts the long-term variance v_bar from the option DataFrame by selecting 
    #the ATM strike at the longest maturity.
    
    #Parameters:
    #- market_prices: DataFrame containing columns ['strike', 'T', 'IV']
    #- S_0: Current stock price (float)
    #- r: Risk-free rate (float, optional, used to calculate Forward price)
    
    #Returns:
    #- v_bar: Long-term variance as a scalar float
    
    #Filter for options with the longest maturity (max T)
    max_T = imp_vol_surface['T'].max()
    longest_options = imp_vol_surface[imp_vol_surface['T'] == max_T]
    
    #Compute Forward price F_0 = S_0 * exp(r * T) for accurate ATM strike selection
    F_0 = S_0 * np.exp(r * max_T)
    
    #Find the single row index closest to the ATM forward strike
    closest_pos = np.abs(longest_options['Strike'].values - F_0).argmin()
    
    #Extract IV 
    sigma_atm = float(longest_options['IV'].iloc[closest_pos])
    
    #Calculate v_bar and force return as a scalar float
    v_bar = float(sigma_atm**2)
    
    return v_bar

#extract_v_bar returns v_bar

v_bar= extract_v_bar(imp_vol_surface_calls, underlying_price)


#Define Loss function

# Fully Vectorized Loss Function
def Heston_loss(params, market_prices, strikes, maturities, r, t_0, S_0, kappa, option_type):
    v_0, v_bar, gamma, rho_xv = params

    # Vectorized evaluation of all options simultaneously
    model_prices = COS_method_Heston(
        r, maturities, t_0, S_0, strikes, v_0, v_bar, gamma, kappa, rho_xv, option_type, L=10, N_k=160
    )
    
    model_prices = np.ravel(model_prices)
    
    # Floor denominator at $2.00 to prevent OTM penny options from dominating
    #weight through vega
    #weights = np.maximum(S_0*np.sqrt(maturities)*0.3989, 2.0)
    weights = np.maximum(market_prices, 2.0)
    
    relative_errors = (model_prices - market_prices) / weights

    if not np.all(np.isfinite(relative_errors)):
        return 1e30

    # Mean Squared Relative Error (MSE) keeps loss values bounded and interpretable
    loss = np.mean(relative_errors**2)
    return loss if np.isfinite(loss) else 1e30


# Updated Optimizer with Expanded Bounds
def Heston_optimizer(
    market_prices,
    strikes,
    maturities,
    underlying_price,
    option_type,
    maxiter=100,
    popsize=10,
    max_points=250,
):
    # Adjusted bounds to prevent artificial boundary clipping
    bounds = [
        (0.005, 0.25),   # v_0: Spot variance (7% to 50% vol)
        (0.005, 0.25),   # v_bar: Long-term variance (7% to 50% vol)
        (0.05, 1.50),    # gamma: Expanded upper bound to prevent hitting 1.0 ceiling
        (-0.99, -0.05)   # rho_xv: Enforces realistic negative equity correlation
    ]

    kappa_fixed = 2.0

    market_prices = np.asarray(market_prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    # Filter out penny options and extreme OTM/ITM strikes before optimization
    moneyness = strikes / underlying_price
    valid = (
        np.isfinite(market_prices)
        & np.isfinite(strikes)
        & np.isfinite(maturities)
        & (market_prices >= 1.0)       # Exclude contracts worth < $1.00
        & (maturities > 0.01)
        & (moneyness >= 0.75)          # Keep strikes between 75% and 125%
        & (moneyness <= 1.25)
    )
    
    market_prices = market_prices[valid]
    strikes = strikes[valid]
    maturities = maturities[valid]

    if len(market_prices) == 0:
        raise ValueError("No valid market prices remain after filtering.")

    if len(market_prices) > max_points:
        selected = np.linspace(0, len(market_prices) - 1, max_points, dtype=int)
        market_prices = market_prices[selected]
        strikes = strikes[selected]
        maturities = maturities[selected]

    S_0 = underlying_price
    t_0 = 0.0

    res = differential_evolution(
        Heston_loss, 
        bounds=bounds, 
        args=(market_prices, strikes, maturities, r_sp500, t_0, S_0, kappa_fixed, option_type),
        strategy="best1bin", 
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        seed=42,
        workers=1,
    )
    return res

    

    

###############################################
#Test for Heston model
#print(v_bar)

#explain the bounds in the optimizer
#maybe change extract_v_bar in order to show it can be call or put
#finish defining the market_prices etc
# imp_vol_surface gives IV, K, T, need still price, 
#extract_v_bar argument needs to be changed

print("Starting Heston calibration...", flush=True)
result = Heston_optimizer(
    imp_vol_surface_calls["Market_Price"].to_numpy(),
    imp_vol_surface_calls["Strike"].to_numpy(),
    imp_vol_surface_calls["T"].to_numpy(),
    underlying_price,
    "call",
    maxiter=100,
    popsize=10,
    max_points=250,
)
print("Heston calibration finished.", flush=True)
v_0_opt, v_bar_opt, gamma_opt, rho_xv_opt =result.x
print('The optimal parameters are:', flush=True)
print(f'v_0: {v_0_opt} with bounds (0.005, 0.25)', flush=True)
print(f'v_bar: {v_bar_opt} with bounds (0.005, 0.25)', flush=True)
print(f'gamma: {gamma_opt} with bounds (0.05, 1.50)', flush=True)
print(f'rho_xv: {rho_xv_opt} with bounds (-0.99, -0.05)', flush=True)
print(f'Objective function value: {result.fun}', flush=True)




###########################################################################
#Analysis of Heston model fit
######################################################################
#The MSE, MSPE of the Heston fit
prices_Heston_fit= COS_method_Heston(r_sp500, 
                       imp_vol_surface_calls["T"].to_numpy(), 
                       0, underlying_price, 
                       imp_vol_surface_calls["Strike"].to_numpy(), 
                       v_0_opt, v_bar_opt, gamma_opt, kappa_fixed, 
                       rho_xv_opt, "call", L=10, N_k=160)


market_prices_fit = imp_vol_surface_calls["Market_Price"].to_numpy(dtype=float)
price_errors_fit = prices_Heston_fit - market_prices_fit
MSPE_Heston_fit_points = 100 * (
    price_errors_fit / np.maximum(market_prices_fit, 2)
) ** 2


#Lets make a df to store the results
df_prices_Heston_fit= pd.DataFrame({
    "Strike": imp_vol_surface_calls["Strike"].to_numpy(),
    "T": imp_vol_surface_calls["T"].to_numpy(),
    "Moneyness": imp_vol_surface_calls["Strike"].to_numpy()/underlying_price,
    "Market_Price": market_prices_fit,
    "Heston_Fit_Price": prices_Heston_fit,
    "MSPE_Contribution": MSPE_Heston_fit_points
})


#Plot the per-contract squared relative pricing error contributions
Hest_fit_error_grid = df_prices_Heston_fit.pivot_table(
    index="T",
    columns="Moneyness",
    values="MSPE_Contribution",
    aggfunc="mean",
)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    Hest_fit_error_grid, 
    annot=False,
    xticklabels=False,
    yticklabels=False,
    cmap="YlOrRd", 
    ax=ax,
    cbar_kws={"label": "Percentage squared error "},
)
max_tick_labels = 10
x_tick_indices = np.unique(np.linspace(
    0, len(Hest_fit_error_grid.columns) - 1,
    min(max_tick_labels, len(Hest_fit_error_grid.columns)),
    dtype=int,
))
y_tick_indices = np.unique(np.linspace(
    0, len(Hest_fit_error_grid.index) - 1,
    min(max_tick_labels, len(Hest_fit_error_grid.index)),
    dtype=int,
))
ax.set_xticks(x_tick_indices + 0.5)
moneyness_values = [
    float(Hest_fit_error_grid.columns[i]) for i in x_tick_indices
]
moneyness_precision = 2
moneyness_labels = [
    f"{moneyness:.{moneyness_precision}f}" for moneyness in moneyness_values
]
while len(set(moneyness_labels)) < len(moneyness_labels) and moneyness_precision < 10:
    moneyness_precision += 1
    moneyness_labels = [
        f"{moneyness:.{moneyness_precision}f}" for moneyness in moneyness_values
    ]
if len(set(moneyness_labels)) < len(moneyness_labels):
    moneyness_labels = [repr(moneyness) for moneyness in moneyness_values]
ax.set_xticklabels(moneyness_labels, rotation=45, ha="right")
maturity_values = [float(Hest_fit_error_grid.index[i]) for i in y_tick_indices]
maturity_precision = 2
maturity_labels = [
    f"{maturity:.{maturity_precision}f}" for maturity in maturity_values
]
while len(set(maturity_labels)) < len(maturity_labels) and maturity_precision < 10:
    maturity_precision += 1
    maturity_labels = [
        f"{maturity:.{maturity_precision}f}" for maturity in maturity_values
    ]
if len(set(maturity_labels)) < len(maturity_labels):
    maturity_labels = [repr(maturity) for maturity in maturity_values]
ax.set_yticks(y_tick_indices + 0.5)
ax.set_yticklabels(maturity_labels, rotation=0)
ax.set_title("Heston Model Squared Relative Pricing Error")
ax.set_xlabel("Moneyness")
ax.set_ylabel("Maturity (Years)")
ax.invert_yaxis()

fig.tight_layout()
fig.savefig("figures/error_heatmap.png", dpi=300)
plt.show()


MSE_Heston_fit = np.mean(price_errors_fit**2)
MSPE_Heston_fit = np.mean(MSPE_Heston_fit_points)

print(f'MSE of Heston market prices fit: {MSE_Heston_fit}', flush=True)
print(f'MSPE of Heston market prices fit: {MSPE_Heston_fit} %', flush=True)


#Now calculate the implied volatility of the Heston fit using the BS_NR_Vect function
imp_vol_heston_fit= BS_NR_Vect(prices_Heston_fit, Call_price_FK, 
                               r_sp500, 0.2, 
                               imp_vol_surface_calls["T"].to_numpy(), 
                               0, underlying_price, 
                               imp_vol_surface_calls["Strike"].to_numpy(), 
                               10**(-4), 10**5, q_sp500)


#define function that will calc the MSE and MSPE of the Heston fit in terms of implied volatility
#we have to watch out for Nan values
def calculate_heston_fit_errors(imp_vol_heston_fit, imp_vol_surface_calls, plot=False):

    #extract pure 1D NumPy arrays
    y_pred = np.asarray(imp_vol_heston_fit, dtype=float).ravel()
    y_true = imp_vol_surface_calls["IV"].to_numpy(dtype=float).ravel()

    #find mutual valid (non-NaN) mask
    not_nan_indices = ~np.isnan(y_pred) & ~np.isnan(y_true)
    total_nan_count = np.sum(~not_nan_indices)
    #filter arrays down to valid entries
    y_pred_clean = y_pred[not_nan_indices]
    y_true_clean = y_true[not_nan_indices]

    #calculate Mean Squared Error (MSE)
    SE_IV_Heston_fit = (y_pred_clean - y_true_clean) ** 2
    MSE_IV_Heston_fit = np.mean(SE_IV_Heston_fit)

    #calculate Mean Squared Percentage Error (MSPE)
    #Note: Use a small epsilon (e.g., 1e-8 or 0.01) to prevent division by zero instead of 2
    epsilon = 1e-8
    SPE_IV_Heston_fit = ((y_pred_clean - y_true_clean) / np.maximum(y_true_clean, epsilon)) ** 2
    MSPE_IV_Heston_fit = 100 * np.mean(SPE_IV_Heston_fit)


    #plot and show the percentage squarederrors if requested
    if plot:
        plt.figure(figsize=(10, 6))
        #plt.hist(SE_IV_Heston_fit, bins=30, alpha=0.7, color='blue', label='Squared Errors')
        plt.hist(100*SPE_IV_Heston_fit, bins=30, alpha=0.7, color='orange', label='Squared Percentage Errors')
        plt.title('Histogram of Percentage Squared Errors for Heston Implied Volatility Fit')
        plt.xlabel('Percentage Squared Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()    
        plt.show(block=True)



    return MSE_IV_Heston_fit, MSPE_IV_Heston_fit, total_nan_count


MSE_IV_Heston_fit, MSPE_IV_Heston_fit, total_nan_count = calculate_heston_fit_errors(imp_vol_heston_fit, imp_vol_surface_calls, plot=True)
print(f'MSE of Heston implied volatility fit: {MSE_IV_Heston_fit}', flush=True)
print(f'MSPE of Heston implied volatility fit: {MSPE_IV_Heston_fit} %', flush=True)
print(f'Total NaN values: {total_nan_count}', flush=True)
print(f'Total valid values used in error calculation: {len(imp_vol_heston_fit) - total_nan_count}', flush=True)



print("Run complete", flush=True)
