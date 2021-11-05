def get_decoder(option):
    if option['decoder'].lower() == 'rcab':
        from model.decoder.rcab import rcab_decoder
        decoder = rcab_decoder(option)

    return decoder
