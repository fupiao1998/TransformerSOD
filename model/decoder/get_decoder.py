def get_decoder(option):
    if option['decoder'].lower() == 'rcab':
        from model.decoder.rcab_decoder import rcab_decoder
        decoder = rcab_decoder(option)
    elif option['decoder'].lower() == 'simple':
        from model.decoder.simple_decoder import simple_decoder
        decoder = simple_decoder(option)
    elif option['decoder'].lower() == 'cat':
        from model.decoder.concat_decoder import concat_decoder
        decoder = concat_decoder(option)
    elif option['decoder'].lower() == 'cat_deep':
        from model.decoder.concat_decoder_deep_sup import concat_decoder_deep_sup
        decoder = concat_decoder_deep_sup(option)
    elif option['decoder'].lower() == 'trans':
        vit_params = {}
        vit_params['embed_dim'] = option['neck_channel']
        vit_params['depth'] = 4
        vit_params['num_heads'] = 4
        vit_params['mlp_ratio'] = 3.0
        vit_params['hid_dim'] = 64
        vit_params['decoder_feat_HxW'] = 12*12
        from model.decoder.transformer_decoder import transformer_decoder
        decoder = transformer_decoder(vit_params, channels=option['neck_channel'], hid_dim=option['neck_channel'])
        # need to fix the uniform API
    else:
        raise KeyError('No decoder named {}'.format(option['decoder']))

    return decoder
