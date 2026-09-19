from experiments.legacy import atlas_simulator as _m
for _k, _v in vars(_m).items():
    if not _k.startswith('__'):
        globals()[_k] = _v
