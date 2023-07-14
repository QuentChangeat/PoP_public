import numpy as np

def CustomProfile(config, planet=None, star=None, optimizer=None, model=None, observations=None, parser=None, show_params=False, phasecurve=True, renorm=True):
    filepath = config['Observation']['iraclis_file']
    try:
        renorm = config['Observation']['renormalize_errors']
    except:
        pass

    params_obs = parse_config(config)

    # Get the pipelines
    pipeObs = prepare_observation(filepath, planet, star, params_obs)
    #pipeModel = prepare_modelpipe(numramp, planet, star)

    # Create the model
    #from pop.model.pcmodel import PhaseOrbitalModel
    #model = PhaseOrbitalModel(planet=planet,
    #            star = star,
    #            observation=pipeObs,
    #            modelpipe = None,
    #            stellar_grid = 'Phoenix_2012_13',
    #            passbands = 'uniform_phoenix_2012_13')
    #model.build()
    model = prepare_model(planet, star, pipeObs, parser = parser)
    
    ## this is to avoid cataclismic crash if log fit as fp_fs = 0 by default
    
    if show_params:
        from pop.pop import show_parameters
        show_parameters(model, pipeObs)
        return

    # Set the model and observation for the optimizer
    optimizer.set_model(model)
    optimizer.set_observed(pipeObs)

    # Use the default setup for HST White fitting
    optimizer = default_opimization(optimizer)
    # Parse the custom setups from parfile
    parser.setup_optimizer(optimizer)

    sols = optimizer.fit()

    for solution,optimized_map,optimized_value,values in optimizer.get_solution():
        optimizer.update_model(optimized_map)
        result1 = model.model()
        optimizer.update_model(optimized_value)
        result2 = model.model()
        break

    #result = model.model()

    factor = 0.0

    if renorm:
        sp_model = result1[1]
        sp_obs = optimizer._observed.spectrum
        sp_err = optimizer._observed.errorBar
        rms = np.sqrt(np.sum( (sp_model[0] - sp_obs)**2 )/len(sp_obs))
        factor = rms/np.median(sp_err)
        print('Inflation factor is set to:'+str(factor))
        #pipeObs._units[0].error_inflation_factor = factor
        pipeObs = prepare_observation(filepath, planet, star, params_obs)
        pipeObs['sys_inf'] = factor
        model = prepare_model(planet, star, pipeObs, parser = parser)
        optimizer.set_model(model)
        optimizer.set_observed(pipeObs)
        optimizer = default_opimization(optimizer)
        parser.setup_optimizer(optimizer)

        sols = optimizer.fit()
        for solution,optimized_map,optimized_value,values in optimizer.get_solution():
            optimizer.update_model(optimized_map)
            result1 = model.model()
            optimizer.update_model(optimized_value)
            result2 = model.model()
            break

        #result = model.model()

    out_dico ={}
    out_dico['Results'] = {}
    out_dico['Observation'] = {}
    out_dico['Results'] = save_results_to_dict(out_dico['Results'], result1, sols, save_sols=True, res_label='_map')
    out_dico['Observation'] = save_observation_to_dict(out_dico['Observation'], pipeObs, sols)
    out_dico['Results'] = save_results_to_dict(out_dico['Results'], result2, sols, save_sols=False, res_label='_value')
    #out_dico['Observation']['factor_inf'] = factor

    return optimizer, sols, out_dico

def parse_config(config):

    remove_orb = config['Observation']['num_removed_orbits']
    remove_exp = config['Observation']['num_removed_exposures']
    
    expfactorAF = config['Observation']['expfactorAF']
    expfactorBF = config['Observation']['expfactorBF']
    expfactorAR = config['Observation']['expfactorAR']
    expfactorBR = config['Observation']['expfactorBR']

    linfactorF = config['Observation']['linfactorF']
    quadfactorF = config['Observation']['quadfactorF']
    normF = config['Observation']['normF']
    linfactorR = config['Observation']['linfactorR']
    quadfactorR = config['Observation']['quadfactorR']
    normR = config['Observation']['normR']
    segment_times = config['Observation']['segment_times']
    visit_times = config['Observation']['visit_times']

    param_obs = [remove_orb, remove_exp, 
                    expfactorAF, expfactorBF, expfactorAR, expfactorBR,
                    linfactorF, quadfactorF, normF, linfactorR, quadfactorR, normR,
                    segment_times, visit_times]
    return param_obs

def prepare_model(planet, star, pipeObs, parser, pipeModel=None):
    #from pop.model import PhaseOrbitalModel
    #model = PhaseOrbitalModel(planet=planet,
    #                star = star,
    #                observation=pipeObs)
    from pop.pipeline import ObservationPipeline
    from pop.pipeline.units import IamUselessUnit

    param_model = parser._raw_config.dict()['Overide']

    if pipeModel is None:
        units = [IamUselessUnit(name='useless'),]
        order = ['useless',]
        pipeModel = ObservationPipeline(units, order, planet, star)

    model = parser.generate_model(planet=planet, star=star, obs=pipeObs, modelpipe=pipeModel)
    model.build()
    ## this is to avoid cataclismic crash if log fit as fp_fs = 0 by default
    model['fp_fs_1'] = 1e-15
    try:
        model['rp_rs_1'] = 1e-15
    except:
        model['rp_rs2_1'] = 0.015
    #model['rp_rs_1'] = 1e-30
    for p in param_model:
        try:
            model[p] = param_model[p]
        except:
            pass
    return model

def prepare_observation(filepath, planet, star, params=[], idx='white'):
    from pop.pipeline import ObservationPipeline
    from pop.pipeline.units.hst import IraclisLightLoader, HSTOrbitFilter, HSTObservationPhaseRampsUnit, HSTFlatten

    remove_orb, remove_exp, expfactorAF, expfactorBF, expfactorAR, expfactorBR, linfactorF, quadfactorF, normF, linfactorR, quadfactorR, normR, segment_times, visit_times = params

    ir = IraclisLightLoader(name='iraclis_loader', file_path=filepath)
    fi = HSTOrbitFilter(name='filter', remove_orbits=remove_orb, remove_first_exposures=remove_exp)
    ra = HSTObservationPhaseRampsUnit(name='sys',
                                    expfactorAF=expfactorAF, expfactorBF=expfactorBF,
                                    expfactorAR=expfactorAR, expfactorBR=expfactorBR,
                                    linfactorF=linfactorF, quadfactorF = quadfactorF, normF=normF,
                                    linfactorR=linfactorR, quadfactorR= quadfactorR, normR=normR,
                                    segment_times=segment_times, visit_times=visit_times)
    fl = HSTFlatten(name = 'flatten')
    pipeObs = ObservationPipeline(units=[ir,fi, ra,fl], 
                                  order=['iraclis_loader','filter', 'sys', 'flatten'], 
                                  planet=planet, star=star)
    pipeObs.build()
    ir.set_spectrum(idx = idx)
    A = pipeObs.run()

    return pipeObs

#def prepare_modelpipe(numramp, planet, star):
#    from pop.pipeline import ObservationPipeline
#    from pop.pipeline.units import HSTRampsUnit, NormalizingUnit
    
#    hstRamp = HSTRampsUnit('hstramp', num_exp_ramp=numramp)
#    hstNorm = NormalizingUnit('hstnorm')
#    pipeModel = ObservationPipeline(units=[hstRamp, hstNorm], order=['hstnorm','hstramp',], planet=planet, star=star)
    
#    return pipeModel

def default_opimization(optimizer):

    optimizer.disable_fit('planet_radius')

    optimizer.disable_fit('fp_fs_1')
    optimizer.set_boundary('fp_fs_1',[1e-5,1e-2])
    optimizer.set_mode('fp_fs_1','log')
    try:
        optimizer.disable_fit('rp_rs_1')
        optimizer.set_boundary('rp_rs_1',[1e-3,0.5])
        optimizer.set_mode('rp_rs_1','log')
    except:
        optimizer.disable_fit('rp_rs2_1')
        optimizer.set_boundary('rp_rs2_1',[1e-7,0.1])
        optimizer.set_mode('rp_rs2_1','log')

    
    mid_min = optimizer._model['mid_time']-0.1
    mid_max = optimizer._model['mid_time']+0.1
    optimizer.enable_fit('mid_time')
    optimizer.set_boundary('mid_time',[mid_min,mid_max])
    
    import numpy as np
    off = np.median(optimizer._observed.spectrum)
    #optimizer._observed['sys_normF'] = off
    #optimizer._observed['sys_normR'] = off
    #bounds = [10**(np.log10(off)-1), 10**(np.log10(off)+1)]

    #optimizer.enable_fit('sys_normF')
    #optimizer.set_boundary('sys_normF',bounds)
    #optimizer.set_mode('sys_normF','log')

    #optimizer.enable_fit('sys_normR')
    #optimizer.set_boundary('sys_normR',bounds)
    #optimizer.set_mode('sys_normR','log')

    #optimizer.enable_fit('sys_expAF')
    #optimizer.set_boundary('sys_expAF',[1e-5,1e-1])
    #optimizer.set_mode('sys_expAF','log')

    #optimizer.enable_fit('sys_expAR')
    #optimizer.set_boundary('sys_expAR',[1e-5,1e-1])
    #optimizer.set_mode('sys_expAR','log')

    #optimizer.enable_fit('sys_expBF')
    #optimizer.set_boundary('sys_expBF',[1,1e4])
    #optimizer.set_mode('sys_expBF','log')

    #optimizer.enable_fit('sys_expBR')
    #optimizer.set_boundary('sys_expBR',[1,1e4])
    #optimizer.set_mode('sys_expBR','log')

    #optimizer.enable_fit('sys_linF')
    #optimizer.set_boundary('sys_linF',[-0.02,0.02])
    #optimizer.set_mode('sys_linF','linear')
    
    #optimizer.enable_fit('sys_linR')
    #optimizer.set_boundary('sys_linR',[-0.02,0.02])
    #optimizer.set_mode('sys_linR','linear')

    return optimizer

def save_results_to_dict(dico, result, sols, save_sols=True, res_label=''):
    dico['timelist'+res_label] = result[0]
    dico['lightcurve'+res_label] = result[1]
    dico['wavelist'+res_label] = result[2]
    if save_sols:
        dico['tracedata'] = sols['solution0']['tracedata']
        dico['weights'] = sols['solution0']['weights']
        dico['fit_params'] = sols['solution0']['fit_params']
        dico['Statistics'] = sols['solution0']['Statistics']
    return dico

def save_observation_to_dict(dico, observation, sols):
    dico = {'timelist': observation.timelist,
                                'lightcurve': observation.shapedSpectrum,
                                'error': observation.shapedErrorBar,
                                'wavelist': observation.wavelist}
    return dico
