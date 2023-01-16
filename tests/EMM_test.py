from utils.math_functions import np, minimize, snorm
from utils.plotting_functions import plot, plt
import pandas as pd


def plot_simulated(obs_data, vol_data, T_horizon):
    time = np.linspace(0., T_horizon, T_horizon)
    fig, ax = plt.subplots(2, 1)
    plot(time, lines=[obs_data], label_args=["Price Process"], xlabel="Time", ylabel="Price Process",
         title="GARCH(1,1) Model", fig=fig, ax=ax[0])
    plot(time, lines=[vol_data], label_args=["Volatility Process"], xlabel="Time", ylabel="Volatility Process",
         title="GARCH(1,1) Model", fig=fig, ax=ax[1])
    #plt.close()


def true_model(ht1, a=-0.736, b=0.9, s=0.363):
    assert (-1 <= b <= 1 and s > 0)
    ht = a + b * ht1 + s * np.random.normal()  # log(sigma_{t}^{2})
    obs = np.exp(0.5 * ht) * np.random.normal()  # y_{t}
    return [obs, ht]


def aux_model(st1, obst1, w=2.0, a=0.4, b=0.1):
    """ GARCH(1,1) auxiliary model """
    assert (w > 0 and a >= 0 and b >= 0 and a + b < 1)
    st = w + a * obst1 ** 2 + b * st1  # sigma_{t}^{2}
    obst = np.power(st, 0.5) * np.random.normal()  # y_{t}
    return [obst, st]


def simulate_true(T_horizon, a=-0.736, b=0.9, s=0.363):
    volData = [a / (1 - b)]  # Use long term variance as initial vol estimate (since no information)
    obsData = [np.random.normal(loc=0., scale=np.exp(0.5 * volData[0]))]  # Directly from model
    for i in range(T_horizon):
        new_data = true_model(volData[i], a=a, b=b, s=s)
        obsData.append(new_data[0])
        volData.append(new_data[1])
    return pd.DataFrame(data={'price': obsData, 'vol2': np.exp(volData)})


def simulate_aux(T_horizon, w=2., a=0.4, b=0.1):
    volData = [w / (1 - a - b)]  # Use long term variance as initial vol estimate (since no information)
    obsData = [np.random.normal(loc=0., scale=np.exp(0.5 * volData[0]))]  # Directly from model
    for i in range(T_horizon):
        new_data = aux_model(volData[i], obsData[i], w=w, a=a, b=b)
        obsData.append(new_data[0])
        volData.append(new_data[1])
    return pd.DataFrame(data={'price': obsData, 'vol2': volData})


def test_aux_MLE(auxMLE, dataFrame, tol=1e-3):
    prices = dataFrame.loc[:, "price"]
    """ Simulate vols under auxiliary model MLE """
    w, a, b = auxMLE
    aux_mle_vol2s = [w / (1 - a - b)]
    for i in range(1, dataFrame.shape[0]):
        aux_mle_vol2s.append(w + a * prices[i - 1] ** 2 + b * aux_mle_vol2s[i - 1])
    m = moment_function(dataFrame, aux_mle_vol2s)
    return m / (dataFrame.shape[0]) < tol


def aux_MLE_constraints():
    """ Note that constraints must return a real value and formulated as equality or >= """
    return [{'type': 'ineq', 'fun': lambda args: args[0]}, {'type': 'ineq', 'fun': lambda args: args[1]},
            {'type': 'ineq', 'fun': lambda args: args[2]},
            {'type': 'ineq', 'fun': lambda args: -args[1] - args[2] + 1.}]


def aux_MLE_objective(params, dataFrame):
    prices = dataFrame.loc[:, 'price']
    w, a, b = params
    if not (w > 0. and a > 0. and b > 0. and a + b < 1.):
        """ Side-step calculation if parameter regime is incorrect """
        return np.inf
    newVol2s = [w / (1. - a - b)]
    # TODO: Optimize this code
    for i in range(1, dataFrame.shape[0]):
        newVol2s.append(w + a * prices[i - 1] ** 2 + b * newVol2s[i - 1])
    return -1. * np.sum(-np.log(newVol2s) - prices ** 2 / newVol2s)


def aux_MLE(dataFrame):
    """ The larger the dataset, the lower the variability in the parameter estimates"""
    """ Minimize function occasionally returns very wrong values -> why? """
    return minimize(aux_MLE_objective, np.array([0.1, 0.1, 0.1]), args=dataFrame, constraints=aux_MLE_constraints()).x


def covariance_estimation(dataFrame, params):
    T = dataFrame.shape[0]
    V = np.zeros(shape=(3, 3))
    prices = dataFrame.loc[:, 'price']
    # TODO: Optimize this code
    w, a, b = params
    vol2s = [w / (1 - a - b)]
    for i in range(1, T):
        vol2s.append(w + a * prices[i - 1] ** 2 + b * vol2s[i - 1])
    """ Given vols generated under auxMLE, calculate scores """
    for i in range(1, T):
        p2 = prices[i - 1] ** 2
        v = vol2s[i - 1]
        s = np.atleast_2d([1., p2, v]).T
        V += 0.25 * np.power(vol2s[i], -2) * np.power((1. - np.power(prices[i], 2) / vol2s[i]), 2) * s @ s.T
    return V / (T - 1.)


def moment_function(dataFrame, aux_mle_vol2s):
    m = np.zeros(shape=(3, 1))
    prices = dataFrame.loc[:, 'price']
    T = dataFrame.shape[0]
    """ Now use simulated latents to compute score functions """
    for i in range(1, T):
        p2 = prices[i - 1] ** 2
        v = aux_mle_vol2s[i - 1]
        s = np.atleast_2d([1., p2, v]).T
        m += -0.5 * np.power(aux_mle_vol2s[i], -1) * (1. - np.power(prices[i], 2) / aux_mle_vol2s[i]) * s
    return m / (T - 1.)


def true_optimize_constraints():
    return [{'type': 'ineq', 'fun': lambda args: args[2]}, {'type': 'ineq', 'fun': lambda args: args[1] + 1.},
            {'type': 'ineq', 'fun': lambda args: -args[1] + 1.}]


def true_optimize_objective(estParams, args):
    a, b, s = estParams  # For a given parameter theta
    print(a, b, s)
    if not (-1. <= b <= 1. and s > 0):
        return np.inf
    Thorizon = 99999
    df = simulate_true(Thorizon, a, b, s)  # Simulated data from true model under fixed parameters
    prices = df.loc[:, "price"]
    mleParams, invVhat = args[0], args[1]
    """ Simulate vols under auxiliary model MLE """
    w, a, b = mleParams
    aux_mle_vol2s = [w / (1 - a - b)]
    for i in range(1, df.shape[0]):
        aux_mle_vol2s.append(w + a * prices[i - 1] ** 2 + b * aux_mle_vol2s[i - 1])
    m = moment_function(df, aux_mle_vol2s)
    return m.T @ invVhat @ m


def true_optimize(mleParams, invVhat):
    return minimize(true_optimize_objective, np.array([-0.1, 0.5, 0.1]), args=[mleParams, invVhat],
                    constraints=true_optimize_constraints())


def EMM():
    """ Toy problem inspired by https://support.sas.com/rnd/app/ets/examples/emmweb/index.htm"""
    """ If parameters are initialized close to each other, initial guess is always returned -> why? """
    trueDf = simulate_true(9999, a=0.736, b=0.9, s=0.363)
    # If data size is too large, auxMLE is initial guess (degenerate likelihood)
    auxParams = aux_MLE(trueDf)  # Obtain MLE for auxiliary model given true data
    """ Below Passes if correct optimum is found but other times still does not """
    assert (test_aux_MLE(auxParams, trueDf, 1e-3).all())
    print(auxParams)
    invVhat = np.linalg.inv(covariance_estimation(trueDf, auxParams))
    """ This step very slow if the likelihood is multimodal and peaked; might also converge to wrong optimum """
    trueParamE = true_optimize(auxParams, invVhat).x
    return trueParamE


print(EMM())
