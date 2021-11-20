#### Grid Notes
For every environment, different simbench/pandapower energy systems can be
choosen. The difficulty of the learning problem depends mainly on the number of
generators (~number of actuators) and the number of buses (~number of sensors
and ~complexity of the underlying function).

To decide which system to use for experiments, here a quick list with the
relevant information for each simbench system for quick access:

| simbench id       | n_buses   | n_ext_grid    | n_gen     | n_sgen    | n_loads   |
|---|---|---|---|---|---|
| 1-EHV-mixed--0-sw | 3085      | 7             | 338       | 225       | 390       |
| 1-HV-mixed--0-sw  | 306       | 3             | 0         | 103       | 58        |
| 1-HV-urban--0-sw  | 372       | 1             | 0         | 98        | 79        |
| 1-MV-rural--0-sw  | 97        | 1             | 0         | 102       | 96        |
| 1-MV-semiurb--0-sw| 117       | 1             | 0         | 121       | 115       |
| 1-MV-urban--0-sw  | 144       | 1             | 0         | 134       | 139       |
| 1-MV-comm--0-sw   | 107       | 1             | 0         | 89        | 98        |
| 1-LV-rural1--0-sw | 15        | 1             | 0         | 4         | 13        |
| 1-LV-rural2--0-sw | 97        | 1             | 0         | 8         | 99        |
| 1-LV-rural3--0-sw | 129       | 1             | 0         | 17        | 118       |
| 1-LV-semiurb4--0-sw| 44       | 1             | 0         | 1         | 41        |
| 1-LV-semiurb5--0-sw | 111     | 1             | 0         | 9         | 104       |
| 1-LV-urban6--0-sw | 59        | 1             | 0         | 5         | 111       |
| 1-HV-mixed--0-sw  | 306       | 1             | 0         | 103       | 58        |

#### Environments
TODO

### Standard OPF (SimpleOpfEnv)

### Reactive power market (QMarketEnv)

### Economic dispatch (EcoDispatchEnv)

### Multi-agent bidding in reactive power market (BiddingQMarketEnv)



#### How to create a new env
What needs to be done? (action_space, observation_space, sampling, etc)
