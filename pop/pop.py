## Pipeline of Pipes?

def show_parameters(model, pipe):
    import tabulate
    print('')
    print('-----------------------------------------------')
    print('----------Available Model Parameters-----------')
    print('-----------------------------------------------')
    print('')

    keywords = [k for k, v in model.fittingParameters.items()]

    short_desc = []
    for k, v in model.fittingParameters.items():
        doc = v[2].__doc__
        if doc is None or doc == 'None':
            short_desc.append('')
        else:
            split = doc.split('\n')
            for spl in split:
                if len(spl) > 0:
                    s = spl
                    break

            short_desc.append(s)

    output = tabulate.tabulate(zip(keywords,  short_desc),
                               headers=['Param Name', 'Short Desc'],
                               tablefmt="fancy_grid")
    print(output)
    print('\n\n')

    print('')
    print('-----------------------------------------------')
    print('-------Available Pipeline Parameters-----------')
    print('-----------------------------------------------')
    print('')

    keywords = [k for k, v in pipe.fittingParameters.items()]

    short_desc = []
    for k, v in model.fittingParameters.items():
        doc = v[2].__doc__
        if doc is None or doc == 'None':
            short_desc.append('')
        else:
            split = doc.split('\n')
            for spl in split:
                if len(spl) > 0:
                    s = spl
                    break

            short_desc.append(s)

    output = tabulate.tabulate(zip(keywords,  short_desc),
                               headers=['Param Name', 'Short Desc'],
                               tablefmt="fancy_grid")
    print(output)
    print('\n\n')


def main():
    import argparse
    import datetime

    import logging
    from taurex.mpi import get_rank
    from taurex.log import setLogLevel
    from taurex.log.logger import root_logger
    from .parameter.popparser import PopParser
    from taurex.output.hdf5 import HDF5Output
    from taurex.util.output import store_contributions
    from . import __version__ as version

    import numpy as np

    parser = argparse.ArgumentParser(description='PoP {}'.format(version))

    parser.add_argument("-i", "--input", dest='input_file', type=str,
                        help="Input par file to pass")

    #parser.add_argument("-R", "--retrieval", dest='retrieval', default=False,
    #                    help="When set, runs retrieval", action='store_true')

    parser.add_argument("-o", "--output_file", dest='output_file', type=str)

    parser.add_argument("-p", "--fit_profile", dest='fit_profile', type=str, default='None')

    parser.add_argument('-v', "--version", dest='version', default=False,
                        help="Display version", action='store_true')

    ### TO IMPLEMENT LATER
    #parser.add_argument("--plugins", dest='plugins', default=False,
    #                    help="Display plugins", action='store_true')

    parser.add_argument("--fitparams", dest='fitparams', default=False,
                        help="Display available fitting params", 
                        action='store_true')

    
    root_logger.info('')
    root_logger.info('----------------------------------------------------------------')
    root_logger.info('----------Pipeline of Pipes (POP) is a TauREx product-----------')
    root_logger.info('----------------------------------------------------------------')
    root_logger.info('')
    
    
    args = parser.parse_args()

    if args.version:
        print(version)
        return

    #if args.plugins:
    #    show_plugins()
    #    return

    if args.input_file is None:
        print('Fatal: No input file specified.')
        return

    root_logger.info('PoP %s', version)

    root_logger.info('PoP PROGRAM START AT %s', datetime.datetime.now())

    # Parse the input file
    pp = PopParser()
    pp.read(args.input_file)

    # get the objects
    planet = pp.generate_planet()
    star = pp.generate_star()

    import os

    ### ANOTHER WAY TO LOAD CUSTOM PROFILES
    profile = None
    #if args.fit_profile != 'None':
    #    f = str(args.fit_profile).split('.py')
    #    print(f[0])
    #    profile = __import__(f[0])
    if args.fit_profile != 'None':
        import importlib
        spec = importlib.util.spec_from_file_location("CustomProfile", args.fit_profile)
        profile = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile)
    if profile is not None:
        optimizer = pp.generate_optimizer()
        optimizer, sols, out_dico = profile.CustomProfile(pp._raw_config.dict(), planet, star, optimizer, 
                                            parser=pp, show_params= args.fitparams)

    #if os.path.splitext(os.path.basename(args.fit_profile))[1] == 'py':
    #    dirname = os.path.dirname(args.fit_profile)
    #    file_name = os.path.splitext(os.path.basename(args.fit_profile))[0]
    #    import sys
    #    sys.path.insert(1, dirname)
    #    from file_name import CustomProfile
    #    optimizer = pp.generate_optimizer()
    #    optimizer, sols, out_dico = CustomProfile(pp._raw_config.dict(), planet, star, optimizer, 
    #                                        parser=pp, show_params= args.fitparams)
    else:
        # Get the pipeline up
        observation = pp.generate_pipeline()
        observation.build()
        # Generate a model from the input
        model = pp.generate_model(planet=planet, star=star, obs=observation)
        
        # build the model (note it also build the pipeline)
        model.build()
        #print(model.fittingParameters.keys())
        optimizer = pp.generate_optimizer()
        optimizer.set_model(model)
        optimizer.set_observed(observation)
        pp.setup_optimizer(optimizer)

        if args.fitparams:
            show_parameters(model, observation)
            return

        import time
        if observation is None:
            logging.critical('No spectrum is defined!!')
            quit()

        start_time = time.time()
        
        solution = optimizer.fit()

        end_time = time.time()

        root_logger.info('Total Retrieval finish in %s seconds', end_time-start_time)

        for _, optimized, _, _ in optimizer.get_solution():
            optimizer.update_model(optimized)
            break

    if args.output_file:
        from taurex.output.hdf5 import HDF5Output
        with HDF5Output(args.output_file) as o:
            optimizer.write(o)
            o.store_dictionary(out_dico, group_name='Outputs')

if __name__ == "__main__":

    main()