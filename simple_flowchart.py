RADAR

date = RADAR['date']

GF(date)

if GF > 0:
    GF = filter(lat_lon(GF))
    GF = filter_alt(RADAR)

    if (GF >0) & (RADAR>0):
        passages = find_passages(GF)

        for passage in passages:
            RADAR = average_profiles(RADAR)
            RADAR = parabola_fit(RADAR)

            CONJUNCTIONS = extract_conjunction(RADAR,GF)

            CONJUNCTIONS.save()

            'next iteration'
    else:
        'next iteration'
else:
    'next iteration'