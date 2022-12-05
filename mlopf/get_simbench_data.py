import simbench as sb


sb_ids = ('1-EHV-mixed--0-sw',
          '1-HV-mixed--0-sw',
          '1-HV-urban--0-sw',
          '1-MV-rural--0-sw',
          '1-MV-semiurb--0-sw',
          '1-MV-urban--0-sw',
          '1-MV-comm--0-sw',
          '1-LV-rural1--0-sw',
          '1-LV-rural2--0-sw',
          '1-LV-rural3--0-sw',
          '1-LV-semiurb4--0-sw',
          '1-LV-semiurb5--0-sw',
          '1-LV-urban6--0-sw')

for sb_id in sb_ids:
    for scenario in (0, 1, 2):

        sb_id = list(sb_id)
        sb_id[-4] = str(scenario)
        sb_id = ''.join(sb_id)

        net = sb.get_simbench_net(sb_id)
        profiles = sb.get_absolute_values(
            net, profiles_instead_of_study_cases=True)

        max_power = profiles[('sgen', 'p_mw')].max(axis=0)
        n_sgen = sum(max_power != 0)

        print(sb_id, ':')
        print('N bus: ', len(net.bus))
        print('N ext_grid: ', len(net.ext_grid))
        print('N gen: ', len(net.gen))
        print('N sgen: ', len(net.sgen), '/', n_sgen)
        print('N load: ', len(net.load))
        print('')
