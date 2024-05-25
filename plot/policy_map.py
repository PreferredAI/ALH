class DynamicMap:
    _map =  {
    'memTD3': {
        'label': 'ALH-g (our)',
        'color': 'blue',
    },
    'memTD32': {
        'label': 'ALH-a (our)',
        'color': 'navy',
    },
    'PPO': {
        'label': 'PPO',
        'color': 'purple',
    },
    'memPPO': {
        'label': 'ALH+PPO (our)',
        'color': 'brown',
    },
    "PPO_not_hard": {
        'label': 'PPO (ideal sampling)',
        'color': 'olive',
    },
    'TD3_not_hard': {
        'label': 'TD3 (ideal sampling)',
        'color': 'orange',
    },
    'TD3': {
        'label': 'TD3',
        'color': 'red',
    },
    'DDPG': {
        'label': 'DDPG',
        'color': 'orange',
    },
    'MBPO': {
        'label': 'MBPO',
        'color': 'limegreen',
    },
    'memTD32_ab1': {
        'label': 'no detach',
        'color': 'purple',
    },
    'memTD32_ab4': {
        'label': 'no detach',
        'color': 'purple',
    },
    'memTD32_ab2': {
        'label': 'no localization',
        'color': 'brown',
    },
    'memTD32_ab3': {
        'label': 'no adaptation',
        'color': 'olive',
    },
    'memTD3_ab1': {
        'label': 'no detach',
        'color': 'purple',
    },
    'memTD3_ab4': {
        'label': 'no detach',
        'color': 'purple',
    },
    'memTD3_ab2': {
        'label': 'no localization',
        'color': 'brown',
    },
    'memTD3_ab3': {
        'label': 'no adaptation',
        'color': 'olive',
    },
    'BC': {
        'label': 'BC',
        'color': 'green',
    },
    'TD3_BC': {
        'label': 'TD3+BC',
        'color': 'red',
    },
    'CQL': {
        'label': 'CQL',
    },
    'DT': {
        'label': 'DT',
    },
    'APE-V': {
        'label': 'APE-V',
    },
    }

    _contain_get = ['memTD3_ab1', 'memTD3_ab2', 'memTD3_ab3', 'memTD3_ab4']
    def __getitem__(self, item):
        if item in self._map:
            return self._map[item]
        else:
            for term in self._contain_get:
                if term in item:
                    return self._map[term]
        return {
        'label': f'{item} (unknown)',
        'color': 'pink',
            }
    def update(self, another):
        self._map.update(another)
        return self


policy_map = DynamicMap()
