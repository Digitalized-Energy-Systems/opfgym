#### Installation
Clone the repository and run `pip install -e .` within some kind of virtual env

#### Grid Notes
For every environment, different simbench/pandapower energy systems can be
choosen. The difficulty of the learning problem depends mainly on the number of
generators (~number of actuators) and the number of buses (~number of sensors
and ~complexity of the underlying function).

To decide which system to use for experiments, here a quick list with the
relevant information for each simbench system for quick access:
(Insert 1,2,3 for current, future and far future system)

| simbench id       | n_buses   | n_ext_grid    | n_gen     | n_sgen        | n_loads   |
|---|---|---|---|---|---|
| 1-EHV-mixed--<1,2,3>-sw | 3085      | 7             | 338       | 225/233/241 *(225/225/225)   | 390       |
| 1-HV-mixed--<1,2,3>-sw  | 306/355/377       | 3             | 0         | 103/109/124 *(57/63/78) | 58        |
| 1-HV-urban--<1,2,3>-sw  | 372/402/428       | 1             | 0         | 98/101/118 *(42/45/62)  | 79        |
| 1-MV-rural--<1,2,3>-sw  | 97/99/99        | 1             | 0         | 102       | 96        |
| 1-MV-semiurb--<1,2,3>-sw| 117/122/122       | 1             | 0         | 121/123/123       | 115/118/122       |
| 1-MV-urban--<1,2,3>-sw  | 144       | 1             | 0         | 134       | 139       |
| 1-MV-comm--<1,2,3>-sw   | 107/110/111       | 1             | 0         | 89/90/90 *(89/89/89)       | 98/98/106        |
| 1-LV-rural1--<1,2,3>-sw | 15        | 1             | 0         | 4/8/8         | 13/14/28        |
| 1-LV-rural2--<1,2,3>-sw | 97        | 1             | 0         | 8/9/11         | 99/103/118        |
| 1-LV-rural3--<1,2,3>-sw | 129       | 1             | 0         | 17/25/27  | 118/145/153 |
| 1-LV-semiurb4--<1,2,3>-sw| 44       | 1             | 0         | 1/1/6         | 41/44/58        |
| 1-LV-semiurb5--<1,2,3>-sw | 111     | 1             | 0         | 9/14/15         | 104/118/129       |
| 1-LV-urban6--<1,2,3>-sw | 59        | 1             | 0         | 5/7/12         | 111/112/135       |

Asterisk: Generators with non-zero active power (reason for zero power unknown)


#### Environments
TODO

### Standard OPF (SimpleOpfEnv)

### Reactive power market (QMarketEnv)

### Economic dispatch (EcoDispatchEnv)

### Multi-agent bidding in reactive power market (BiddingQMarketEnv)



#### How to create a new env
What needs to be done? (action_space, observation_space, sampling, etc)
