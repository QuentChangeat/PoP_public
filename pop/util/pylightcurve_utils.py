## THIS CODE IS COPIED DIRECTLY FROM THE PUBLIC VERSION OF PYLIGHTCURVE:
# https://github.com/ucl-exoplanets/pylightcurve
#
# 
#

import numpy as np
import warnings
from scipy.optimize import curve_fit as scipy_curve_fit

def planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array, ww=0):

    inclination = inclination * np.pi / 180.0
    periastron = periastron * np.pi / 180.0
    ww = ww * np.pi / 180.0

    if eccentricity == 0 and ww == 0:
        vv = 2 * np.pi * (time_array - mid_time) / period
        bb = sma_over_rs * np.cos(vv)
        return [bb * np.sin(inclination), sma_over_rs * np.sin(vv), - bb * np.cos(inclination)]

    if periastron < np.pi / 2:
        aa = 1.0 * np.pi / 2 - periastron
    else:
        aa = 5.0 * np.pi / 2 - periastron
    bb = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(aa / 2))
    if bb < 0:
        bb += 2 * np.pi
    mid_time = float(mid_time) - (period / 2.0 / np.pi) * (bb - eccentricity * np.sin(bb))
    m = (time_array - mid_time - np.int_((time_array - mid_time) / period) * period) * 2.0 * np.pi / period
    u0 = m
    stop = False
    u1 = 0
    for ii in range(10000):  # setting a limit of 1k iterations - arbitrary limit
        u1 = u0 - (u0 - eccentricity * np.sin(u0) - m) / (1 - eccentricity * np.cos(u0))
        stop = (np.abs(u1 - u0) < 10 ** (-7)).all()
        if stop:
            break
        else:
            u0 = u1
    if not stop:
        raise RuntimeError('Failed to find a solution in 10000 loops')

    vv = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(u1 / 2))
    #
    rr = sma_over_rs * (1 - (eccentricity ** 2)) / (np.ones_like(vv) + eccentricity * np.cos(vv))
    aa = np.cos(vv + periastron)
    bb = np.sin(vv + periastron)
    x = rr * bb * np.sin(inclination)
    y = rr * (-aa * np.cos(ww) + bb * np.sin(ww) * np.cos(inclination))
    z = rr * (-aa * np.sin(ww) - bb * np.cos(ww) * np.cos(inclination))

    return [x, y, z]

def eclipse_mid_time(period, sma_over_rs, eccentricity, inclination, periastron, mid_time):
    test_array = np.arange(0, period, 0.001)
    xx, yy, zz = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                              test_array + mid_time)

    test1 = np.where(xx < 0)
    yy = yy[test1]
    test_array = test_array[test1]

    aprox = test_array[np.argmin(np.abs(yy))]

    def function_to_fit(x, t):
        xx, yy, zz = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                            np.array(mid_time + t))
        return yy

    popt, pcov = curve_fit(function_to_fit, [0], [0], p0=[aprox])

    return mid_time + popt[0]


def curve_fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Covariance of the parameters could not be estimated')
        return scipy_curve_fit(*args, **kwargs)

def eclipse(fp_over_fs, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
            time_array, precision=3):

    position_vector = planet_orbit(period, sma_over_rs / rp_over_rs, eccentricity, inclination, periastron + 180,
                                   mid_time, time_array)

    projected_distance = np.where(
        position_vector[0] < 0, 1.0 + 5.0 / rp_over_rs,
        np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return (1.0 + fp_over_fs * transit_flux_drop([0, 0, 0, 0], 1 / rp_over_rs, projected_distance,
                                                 precision=precision, method='zero')) / (1.0 + fp_over_fs)

def transit(limb_darkening_coefficients, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron,
            mid_time, time_array, method='claret', precision=3):

    position_vector = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    projected_distance = np.where(
        position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
        np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return transit_flux_drop(limb_darkening_coefficients, rp_over_rs, projected_distance,
                             method=method, precision=precision)


def transit_flux_drop(limb_darkening_coefficients, rp_over_rs, z_over_rs, method='claret', precision=3):

    z_over_rs = np.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)
    z_over_rs = np.maximum(z_over_rs, 10**(-10))

    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2
    case0 = np.where((z_over_rs == 0) & (rp_over_rs <= 1))
    case1 = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
    casea = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
    caseb = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
    case2 = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
    casec = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
    case3 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
    case4 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
    case5 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
    case6 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
    case7 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))
    plus_case = np.concatenate((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
    minus_case = np.concatenate((case3[0], case4[0], case5[0], case6[0], case7[0]))
    star_case = np.concatenate((case5[0], case6[0], case7[0], casea[0], casec[0]))

    # cross points
    ph = np.arccos(np.clip((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
    theta_1 = np.zeros(len(z_over_rs))
    ph_case = np.concatenate((case5[0], casea[0], casec[0]))
    theta_1[ph_case] = ph[ph_case]
    theta_2 = np.arcsin(np.minimum(rp_over_rs / z_over_rs, 1))
    theta_2[case1] = np.pi
    theta_2[case2] = np.pi / 2.0
    theta_2[casea] = np.pi
    theta_2[casec] = np.pi / 2.0
    theta_2[case7] = ph[case7]

    # flux_upper
    plusflux = np.zeros(len(z_over_rs))
    plusflux[plus_case] = integral_plus_core(method, limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
                                             theta_1[plus_case], theta_2[plus_case], precision=precision)
    if len(case0[0]) > 0:
        plusflux[case0] = integral_centred(method, limb_darkening_coefficients, rp_over_rs, 0.0, np.pi)
    if len(caseb[0]) > 0:
        plusflux[caseb] = integral_centred(method, limb_darkening_coefficients, 1, 0.0, np.pi)

    # flux_lower
    minsflux = np.zeros(len(z_over_rs))
    minsflux[minus_case] = integral_minus_core(method, limb_darkening_coefficients, rp_over_rs,
                                               z_over_rs[minus_case], 0.0, theta_2[minus_case], precision=precision)

    # flux_star
    starflux = np.zeros(len(z_over_rs))
    starflux[star_case] = integral_centred(method, limb_darkening_coefficients, 1, 0.0, ph[star_case])

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, 1, 0.0, 2.0 * np.pi)

    return 1 - (2.0 / total_flux) * (plusflux + starflux - minsflux)


def integral_centred(method, limb_darkening_coefficients, rprs, ww1, ww2):
    return (integral_r[method](limb_darkening_coefficients, rprs)
            - integral_r[method](limb_darkening_coefficients, 0.0)) * np.abs(ww2 - ww1)

def integral_plus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * np.cos(ww1) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
    rr1 = np.clip(rr1, 0, 1)
    rr2 = z * np.cos(ww2) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
    rr2 = np.clip(rr2, 0, 1)
    w1 = np.minimum(ww1, ww2)
    r1 = np.minimum(rr1, rr2)
    w2 = np.maximum(ww1, ww2)
    r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd

def integral_minus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    if len(z) == 0:
        return z
    rr1 = z * np.cos(ww1) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
    rr1 = np.clip(rr1, 0, 1)
    rr2 = z * np.cos(ww2) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
    rr2 = np.clip(rr2, 0, 1)
    w1 = np.minimum(ww1, ww2)
    r1 = np.minimum(rr1, rr2)
    w2 = np.maximum(ww1, ww2)
    r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd




def integral_r_claret(limb_darkening_coefficients, r):
    a1, a2, a3, a4 = limb_darkening_coefficients
    mu44 = 1.0 - r * r
    mu24 = np.sqrt(mu44)
    mu14 = np.sqrt(mu24)
    return - (2.0 * (1.0 - a1 - a2 - a3 - a4) / 4) * mu44 \
           - (2.0 * a1 / 5) * mu44 * mu14 \
           - (2.0 * a2 / 6) * mu44 * mu24 \
           - (2.0 * a3 / 7) * mu44 * mu24 * mu14 \
           - (2.0 * a4 / 8) * mu44 * mu44


def num_claret(r, limb_darkening_coefficients, rprs, z):
    a1, a2, a3, a4 = limb_darkening_coefficients
    rsq = r * r
    mu44 = 1.0 - rsq
    mu24 = np.sqrt(mu44)
    mu14 = np.sqrt(mu24)
    return ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_claret, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for zero method


def integral_r_zero(limb_darkening_coefficients, r):
    musq = 1 - r * r
    return (-1.0 / 6) * musq * 3.0


def num_zero(r, limb_darkening_coefficients, rprs, z):
    rsq = r * r
    return r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_zero(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_zero, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for linear method


def integral_r_linear(limb_darkening_coefficients, r):
    a1 = limb_darkening_coefficients[0]
    musq = 1 - r * r
    return (-1.0 / 6) * musq * (3.0 + a1 * (-3.0 + 2.0 * np.sqrt(musq)))


def num_linear(r, limb_darkening_coefficients, rprs, z):
    a1 = limb_darkening_coefficients[0]
    rsq = r * r
    return (1.0 - a1 * (1.0 - np.sqrt(1.0 - rsq))) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_linear(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_linear, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for quadratic method


def integral_r_quad(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = np.sqrt(musq)
    return (1.0 / 12) * (-4.0 * (a1 + 2.0 * a2) * mu * musq + 6.0 * (-1 + a1 + a2) * musq + 3.0 * a2 * musq * musq)


def num_quad(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    cc = 1.0 - np.sqrt(1.0 - rsq)
    return (1.0 - a1 * cc - a2 * cc * cc) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_quad(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_quad, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for square root method


def integral_r_sqrt(limb_darkening_coefficients, r):
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = np.sqrt(musq)
    return ((-2.0 / 5) * a2 * np.sqrt(mu) - (1.0 / 3) * a1 * mu + (1.0 / 2) * (-1 + a1 + a2)) * musq


def num_sqrt(r, limb_darkening_coefficients, rprs, z):
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    mu = np.sqrt(1.0 - rsq)
    return (1.0 - a1 * (1 - mu) - a2 * (1.0 - np.sqrt(mu))) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_sqrt(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    return gauss_numerical_integration(num_sqrt, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# dictionaries containing the different methods,
# if you define a new method, include the functions in the dictionary as well

integral_r = {
    'claret': integral_r_claret,
    'linear': integral_r_linear,
    'quad': integral_r_quad,
    'sqrt': integral_r_sqrt,
    'zero': integral_r_zero
}

integral_r_f = {
    'claret': integral_r_f_claret,
    'linear': integral_r_f_linear,
    'quad': integral_r_f_quad,
    'sqrt': integral_r_f_sqrt,
    'zero': integral_r_f_zero,
}



# coefficients from https://pomax.github.io/bezierinfo/legendre-gauss.html

gauss0 = [
    [1.0000000000000000, -0.5773502691896257],
    [1.0000000000000000, 0.5773502691896257]
]

gauss10 = [
    [0.2955242247147529, -0.1488743389816312],
    [0.2955242247147529, 0.1488743389816312],
    [0.2692667193099963, -0.4333953941292472],
    [0.2692667193099963, 0.4333953941292472],
    [0.2190863625159820, -0.6794095682990244],
    [0.2190863625159820, 0.6794095682990244],
    [0.1494513491505806, -0.8650633666889845],
    [0.1494513491505806, 0.8650633666889845],
    [0.0666713443086881, -0.9739065285171717],
    [0.0666713443086881, 0.9739065285171717]
]

gauss20 = [
    [0.1527533871307258, -0.0765265211334973],
    [0.1527533871307258, 0.0765265211334973],
    [0.1491729864726037, -0.2277858511416451],
    [0.1491729864726037, 0.2277858511416451],
    [0.1420961093183820, -0.3737060887154195],
    [0.1420961093183820, 0.3737060887154195],
    [0.1316886384491766, -0.5108670019508271],
    [0.1316886384491766, 0.5108670019508271],
    [0.1181945319615184, -0.6360536807265150],
    [0.1181945319615184, 0.6360536807265150],
    [0.1019301198172404, -0.7463319064601508],
    [0.1019301198172404, 0.7463319064601508],
    [0.0832767415767048, -0.8391169718222188],
    [0.0832767415767048, 0.8391169718222188],
    [0.0626720483341091, -0.9122344282513259],
    [0.0626720483341091, 0.9122344282513259],
    [0.0406014298003869, -0.9639719272779138],
    [0.0406014298003869, 0.9639719272779138],
    [0.0176140071391521, -0.9931285991850949],
    [0.0176140071391521, 0.9931285991850949],
]

gauss30 = [
    [0.1028526528935588, -0.0514718425553177],
    [0.1028526528935588, 0.0514718425553177],
    [0.1017623897484055, -0.1538699136085835],
    [0.1017623897484055, 0.1538699136085835],
    [0.0995934205867953, -0.2546369261678899],
    [0.0995934205867953, 0.2546369261678899],
    [0.0963687371746443, -0.3527047255308781],
    [0.0963687371746443, 0.3527047255308781],
    [0.0921225222377861, -0.4470337695380892],
    [0.0921225222377861, 0.4470337695380892],
    [0.0868997872010830, -0.5366241481420199],
    [0.0868997872010830, 0.5366241481420199],
    [0.0807558952294202, -0.6205261829892429],
    [0.0807558952294202, 0.6205261829892429],
    [0.0737559747377052, -0.6978504947933158],
    [0.0737559747377052, 0.6978504947933158],
    [0.0659742298821805, -0.7677774321048262],
    [0.0659742298821805, 0.7677774321048262],
    [0.0574931562176191, -0.8295657623827684],
    [0.0574931562176191, 0.8295657623827684],
    [0.0484026728305941, -0.8825605357920527],
    [0.0484026728305941, 0.8825605357920527],
    [0.0387991925696271, -0.9262000474292743],
    [0.0387991925696271, 0.9262000474292743],
    [0.0287847078833234, -0.9600218649683075],
    [0.0287847078833234, 0.9600218649683075],
    [0.0184664683110910, -0.9836681232797472],
    [0.0184664683110910, 0.9836681232797472],
    [0.0079681924961666, -0.9968934840746495],
    [0.0079681924961666, 0.9968934840746495]
]

gauss40 = [
    [0.0775059479784248, -0.0387724175060508],
    [0.0775059479784248, 0.0387724175060508],
    [0.0770398181642480, -0.1160840706752552],
    [0.0770398181642480, 0.1160840706752552],
    [0.0761103619006262, -0.1926975807013711],
    [0.0761103619006262, 0.1926975807013711],
    [0.0747231690579683, -0.2681521850072537],
    [0.0747231690579683, 0.2681521850072537],
    [0.0728865823958041, -0.3419940908257585],
    [0.0728865823958041, 0.3419940908257585],
    [0.0706116473912868, -0.4137792043716050],
    [0.0706116473912868, 0.4137792043716050],
    [0.0679120458152339, -0.4830758016861787],
    [0.0679120458152339, 0.4830758016861787],
    [0.0648040134566010, -0.5494671250951282],
    [0.0648040134566010, 0.5494671250951282],
    [0.0613062424929289, -0.6125538896679802],
    [0.0613062424929289, 0.6125538896679802],
    [0.0574397690993916, -0.6719566846141796],
    [0.0574397690993916, 0.6719566846141796],
    [0.0532278469839368, -0.7273182551899271],
    [0.0532278469839368, 0.7273182551899271],
    [0.0486958076350722, -0.7783056514265194],
    [0.0486958076350722, 0.7783056514265194],
    [0.0438709081856733, -0.8246122308333117],
    [0.0438709081856733, 0.8246122308333117],
    [0.0387821679744720, -0.8659595032122595],
    [0.0387821679744720, 0.8659595032122595],
    [0.0334601952825478, -0.9020988069688743],
    [0.0334601952825478, 0.9020988069688743],
    [0.0279370069800234, -0.9328128082786765],
    [0.0279370069800234, 0.9328128082786765],
    [0.0222458491941670, -0.9579168192137917],
    [0.0222458491941670, 0.9579168192137917],
    [0.0164210583819079, -0.9772599499837743],
    [0.0164210583819079, 0.9772599499837743],
    [0.0104982845311528, -0.9907262386994570],
    [0.0104982845311528, 0.9907262386994570],
    [0.0045212770985332, -0.9982377097105593],
    [0.0045212770985332, 0.9982377097105593],
]

gauss50 = [
    [0.0621766166553473, -0.0310983383271889],
    [0.0621766166553473, 0.0310983383271889],
    [0.0619360674206832, -0.0931747015600861],
    [0.0619360674206832, 0.0931747015600861],
    [0.0614558995903167, -0.1548905899981459],
    [0.0614558995903167, 0.1548905899981459],
    [0.0607379708417702, -0.2160072368760418],
    [0.0607379708417702, 0.2160072368760418],
    [0.0597850587042655, -0.2762881937795320],
    [0.0597850587042655, 0.2762881937795320],
    [0.0586008498132224, -0.3355002454194373],
    [0.0586008498132224, 0.3355002454194373],
    [0.0571899256477284, -0.3934143118975651],
    [0.0571899256477284, 0.3934143118975651],
    [0.0555577448062125, -0.4498063349740388],
    [0.0555577448062125, 0.4498063349740388],
    [0.0537106218889962, -0.5044581449074642],
    [0.0537106218889962, 0.5044581449074642],
    [0.0516557030695811, -0.5571583045146501],
    [0.0516557030695811, 0.5571583045146501],
    [0.0494009384494663, -0.6077029271849502],
    [0.0494009384494663, 0.6077029271849502],
    [0.0469550513039484, -0.6558964656854394],
    [0.0469550513039484, 0.6558964656854394],
    [0.0443275043388033, -0.7015524687068222],
    [0.0443275043388033, 0.7015524687068222],
    [0.0415284630901477, -0.7444943022260685],
    [0.0415284630901477, 0.7444943022260685],
    [0.0385687566125877, -0.7845558329003993],
    [0.0385687566125877, 0.7845558329003993],
    [0.0354598356151462, -0.8215820708593360],
    [0.0354598356151462, 0.8215820708593360],
    [0.0322137282235780, -0.8554297694299461],
    [0.0322137282235780, 0.8554297694299461],
    [0.0288429935805352, -0.8859679795236131],
    [0.0288429935805352, 0.8859679795236131],
    [0.0253606735700124, -0.9130785566557919],
    [0.0253606735700124, 0.9130785566557919],
    [0.0217802431701248, -0.9366566189448780],
    [0.0217802431701248, 0.9366566189448780],
    [0.0181155607134894, -0.9566109552428079],
    [0.0181155607134894, 0.9566109552428079],
    [0.0143808227614856, -0.9728643851066920],
    [0.0143808227614856, 0.9728643851066920],
    [0.0105905483836510, -0.9853540840480058],
    [0.0105905483836510, 0.9853540840480058],
    [0.0067597991957454, -0.9940319694320907],
    [0.0067597991957454, 0.9940319694320907],
    [0.0029086225531551, -0.9988664044200710],
    [0.0029086225531551, 0.9988664044200710]
]

gauss60 = [
    [0.0519078776312206, -0.0259597723012478],
    [0.0519078776312206, 0.0259597723012478],
    [0.0517679431749102, -0.0778093339495366],
    [0.0517679431749102, 0.0778093339495366],
    [0.0514884515009809, -0.1294491353969450],
    [0.0514884515009809, 0.1294491353969450],
    [0.0510701560698556, -0.1807399648734254],
    [0.0510701560698556, 0.1807399648734254],
    [0.0505141845325094, -0.2315435513760293],
    [0.0505141845325094, 0.2315435513760293],
    [0.0498220356905502, -0.2817229374232617],
    [0.0498220356905502, 0.2817229374232617],
    [0.0489955754557568, -0.3311428482684482],
    [0.0489955754557568, 0.3311428482684482],
    [0.0480370318199712, -0.3796700565767980],
    [0.0480370318199712, 0.3796700565767980],
    [0.0469489888489122, -0.4271737415830784],
    [0.0469489888489122, 0.4271737415830784],
    [0.0457343797161145, -0.4735258417617071],
    [0.0457343797161145, 0.4735258417617071],
    [0.0443964787957871, -0.5186014000585697],
    [0.0443964787957871, 0.5186014000585697],
    [0.0429388928359356, -0.5622789007539445],
    [0.0429388928359356, 0.5622789007539445],
    [0.0413655512355848, -0.6044405970485104],
    [0.0413655512355848, 0.6044405970485104],
    [0.0396806954523808, -0.6449728284894770],
    [0.0396806954523808, 0.6449728284894770],
    [0.0378888675692434, -0.6837663273813555],
    [0.0378888675692434, 0.6837663273813555],
    [0.0359948980510845, -0.7207165133557304],
    [0.0359948980510845, 0.7207165133557304],
    [0.0340038927249464, -0.7557237753065856],
    [0.0340038927249464, 0.7557237753065856],
    [0.0319212190192963, -0.7886937399322641],
    [0.0319212190192963, 0.7886937399322641],
    [0.0297524915007889, -0.8195375261621458],
    [0.0297524915007889, 0.8195375261621458],
    [0.0275035567499248, -0.8481719847859296],
    [0.0275035567499248, 0.8481719847859296],
    [0.0251804776215212, -0.8745199226468983],
    [0.0251804776215212, 0.8745199226468983],
    [0.0227895169439978, -0.8985103108100460],
    [0.0227895169439978, 0.8985103108100460],
    [0.0203371207294573, -0.9200784761776275],
    [0.0203371207294573, 0.9200784761776275],
    [0.0178299010142077, -0.9391662761164232],
    [0.0178299010142077, 0.9391662761164232],
    [0.0152746185967848, -0.9557222558399961],
    [0.0152746185967848, 0.9557222558399961],
    [0.0126781664768160, -0.9697017887650528],
    [0.0126781664768160, 0.9697017887650528],
    [0.0100475571822880, -0.9810672017525982],
    [0.0100475571822880, 0.9810672017525982],
    [0.0073899311633455, -0.9897878952222218],
    [0.0073899311633455, 0.9897878952222218],
    [0.0047127299269536, -0.9958405251188381],
    [0.0047127299269536, 0.9958405251188381],
    [0.0020268119688738, -0.9992101232274361],
    [0.0020268119688738, 0.9992101232274361],
]

gauss_table = [np.swapaxes(gauss0, 0, 1), np.swapaxes(gauss10, 0, 1), np.swapaxes(gauss20, 0, 1),
               np.swapaxes(gauss30, 0, 1), np.swapaxes(gauss40, 0, 1), np.swapaxes(gauss50, 0, 1),
               np.swapaxes(gauss60, 0, 1)]


def gauss_numerical_integration(f, x1, x2, precision, *f_args):

    x1, x2 = (x2 - x1) / 2, (x2 + x1) / 2

    return x1 * np.sum(gauss_table[precision][0][:, None] *
                       f(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)


def sample_function(f, precision=3):

    def sampled_function(x12_array, *args):

        x1_array, x2_array = x12_array

        return gauss_numerical_integration(f, x1_array, x2_array, precision, *list(args))

    return sampled_function
