from pathlib import Path
import numpy as np

def CustomProfile(config, planet=None, star=None, optimizer=None, model=None, observations=None, parser=None, show_params=False, phasecurve=True):
    
    filepath = config['Observation']['iraclis_file']
    div_white = False
    try:
        div_white = config['Observation']['divide_white']
        div_white_file = config['Observation']['white_results']
        div_white_prefix = config['Observation']['white_prefix']
    except:
        pass
    from .HSTWhitePhase_profile import parse_config
    params_obs = parse_config(config)

    renorm = True
    resume = True
    try:
        renorm = config['Observation']['renormalize_errors']
    except:
        pass
    try:
        resume = config['Observation']['resume']
    except:
        pass
    if parser._raw_config.dict()['Optimizer']['optimizer'] == 'multinest':
        root_opt_path = optimizer.dir_multinest
    else:
        root_opt_path = 'optimizer_path'
    
    from pop.model.pcmodel import PhaseOrbitalModel
    from pop.pipeline.units.hst import HSTObservationPhaseRampsUnit
    from .HSTWhitePhase_profile import prepare_model, prepare_observation, default_opimization, save_observation_to_dict, save_results_to_dict

    pipeObs = prepare_observation(filepath, planet, star, params_obs)

    pipeModel = None
    if div_white:
        from pop.pipeline.units.hst import ModelRenormUnit
        from pop.pipeline import ObservationPipeline
        units = [ModelRenormUnit(name='mrenorm', white_res=div_white_file,  res_prefix=div_white_prefix),]
        order = ['mrenorm',]
        pipeModel = ObservationPipeline(units, order, planet, star)

    N_wlbins = len(pipeObs._units[0]._wl)
    
    optimizers = []
    solutions = []
    out_dico = {}

    print('Number of bins to fit is: ', N_wlbins)

    for idx in range(N_wlbins):
        out_dir = root_opt_path+'/bin'+str(idx)
        pipeObs._units[0].set_spectrum(idx = idx, div_white = div_white)
        #pipeObs.build()

        #pipeModel = prepare_modelpipe(numramp, planet, star)
        if resume and Path(out_dir+'/dico.h5').is_file():
            #print('Passing idx: '+str(idx))
            import h5py
            file = h5py.File(out_dir+'/dico.h5')
            temp_dico = file['Outputs']
            sols = None

        else:
            from pop.model.pcmodel import PhaseOrbitalModel
            #model = PhaseOrbitalModel(planet=planet,
            #            star = star,
            #            observation=pipeObs,
            #            modelpipe = None,
            #            stellar_grid = 'Phoenix_2012_13',
            #            passbands = 'uniform_phoenix_2012_13')
            #model.build()
            ## this is to avoid cataclismic crash if log fit as fp_fs = 0 by default
            #model['fp_fs_1'] = 1e-15
            #try:
            #    model['rp_rs_1'] = 1e-15
            #except:
            #    model['rp_rs2_1'] = 0.015

            model = prepare_model(planet, star, pipeObs, parser = parser, pipeModel=pipeModel)

            if show_params:
                from pop.pop import show_parameters
                show_parameters(model, pipeObs)
                return

            optimizer.set_model(model)
            optimizer.set_observed(pipeObs)

            optimizer = default_opimization(optimizer)

            parser.setup_optimizer(optimizer)



            Path(out_dir).mkdir(parents=True, exist_ok=True)
            if parser._raw_config.dict()['Optimizer']['optimizer'] == 'multinest':
                optimizer.dir_multinest = out_dir

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
                pipeObs['sys_inf'] = factor
                sols = optimizer.fit()
                for solution,optimized_map,optimized_value,values in optimizer.get_solution():
                    optimizer.update_model(optimized_map)
                    result1 = model.model()
                    optimizer.update_model(optimized_value)
                    result2 = model.model()
                    break
                #result = model.model()

            from taurex.output.hdf5 import HDF5Output
            with HDF5Output(out_dir+'/dico.h5', append=True) as o:
                optimizer.write(o)
                temp_dico = {}
                temp_dico['Results'+str(idx)] = {}
                temp_dico['Observation'+str(idx)] = {}
                temp_dico['Observation'+str(idx)]['factor_inf'] = factor
                temp_dico['Results'+str(idx)] = save_results_to_dict(temp_dico['Results'+str(idx)], result1, sols, save_sols=True, res_label='_map')
                temp_dico['Results'+str(idx)] = save_results_to_dict(temp_dico['Results'+str(idx)], result2, sols, save_sols=False, res_label='_value')
                temp_dico['Observation'+str(idx)] = save_observation_to_dict(temp_dico['Observation'+str(idx)], pipeObs, sols)
                o.store_dictionary(temp_dico, group_name='Outputs')

        out_dico['Results'+str(idx)] = temp_dico['Results'+str(idx)]
        out_dico['Observation'+str(idx)] = temp_dico['Observation'+str(idx)]
        if div_white:
            temp_white = {}
            temp_white['activated'] = div_white
            temp_white['renorm_t'] = units[0].renorm_t
            temp_white['renorm_s'] = units[0].renorm_sp
            temp_white['renorm_er'] = units[0].renorm_er
            out_dico['div_white'] = temp_white
        
        solutions.append(sols)
        optimizers.append(optimizer)
    
    return optimizer, solutions, out_dico