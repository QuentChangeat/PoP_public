from taurex.parameter import ParameterParser

class PopParser(ParameterParser):

    def __init__(self):
        super().__init__()
        self.collect_unit_classes()
    
    def collect_unit_classes(self):
        from pop.pipeline.units.baseunit import BaseUnit
        from pop.pipeline import units
        self.unit_klasses = set()
        self.unit_klasses.update(self._collect_classes(units, BaseUnit))
    
    def _collect_classes(self, module, base_klass):
        import inspect
        klasses = []
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for _, c in clsmembers:
            if issubclass(c, base_klass) and (c is not base_klass):
                #self.log.debug(f' Found class {c.__name__}')
                klasses.append(c)

        return klasses

    def generate_model(self, chemistry=None, pressure=None,
                       temperature=None, planet=None,
                       star=None, modelpipe=None, obs=None):
        config = self._raw_config.dict()
        if 'Model' in config:
            if planet is None:
                planet = self.generate_planet()
            if star is None:
                star = self.generate_star()
            if modelpipe is None:
                modelpipe = self.generate_pipeline(config['Model'], planet, star)
                config['Model'].pop('order')
            model = self.create_model(config['Model'], planet, star, modelpipe, observation=obs)
        else:
            return None

        return model
    
    def generate_pipeline(self, config=None, planet=None, star=None):
        from taurex.parameter.factory import create_klass
        if config is None:
            config = self._raw_config.dict()['Observation']
        check_key = [k for k, v in config.items() if isinstance(v, dict)]
        units = []
    
        for key in config.keys():
            try:
                kind = config[key]['type']
            except:
                continue
            for klass in self.unit_klasses:
                if kind in klass.input_keywords():
                    config[key].pop('type')
                    config[key]['name'] = key
                    units.append(create_klass(config[key],klass,False))
        
        from pop.pipeline import ObservationPipeline
        try:
            order = config['order']
            pipe = ObservationPipeline(units, order, planet, star)
        except:
            self.warning('No order provided for the Model pipeline, you should create a FAKE one!')
            #from pop.pipeline.units.jwst import JWSTUseless
            #units = [JWSTUseless(name='useless'),]
            #order = ['useless',]
            #pipe = ObservationPipeline(units, order, planet, star)
            raise NotImplementedError('Unit order in Pipe was not provided')
        return pipe
    
    def create_model(self, config, planet, star, modelpipe, observation=None):
        from taurex.model import ForwardModel
        from taurex.parameter.factory import determine_klass, model_factory, get_keywordarg_dict
        config, klass, is_mixin = \
            determine_klass(config, 'model_type',
                            model_factory, ForwardModel)

        kwargs = get_keywordarg_dict(klass, is_mixin)

        if 'planet' in kwargs:
            kwargs['planet'] = planet
        if 'star' in kwargs:
            kwargs['star'] = star
        if 'chemistry' in kwargs:
            kwargs['chemistry'] = gas
        if 'temperature_profile' in kwargs:
            kwargs['temperature_profile'] = temperature
        if 'pressure_profile' in kwargs:
            kwargs['pressure_profile'] = pressure
        if 'modelpipe' in kwargs:
            kwargs['modelpipe'] = modelpipe
        if 'observation' in kwargs:
            kwargs['observation'] = observation
        
        if modelpipe is None:
            mp = self.generate_pipeline(config, planet, star)
            kwargs['modelpipe'] = mp
        kwargs.update(dict([(k, v) for k, v in config.items()
                      if not isinstance(v, dict)]))
        obj = klass(**kwargs)

        #contribs = generate_contributions(config)

        #for c in contribs:
        #    obj.add_contribution(c)

        return obj

    