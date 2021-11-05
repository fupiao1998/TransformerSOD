def get_decoder(option):
    if option['decoder'].lower() == 'rcab':
        from model.decoder.rcab_decoder import rcab_decoder
        decoder = rcab_decoder(option)
    elif option['decoder'].lower() == 'trans':
        vit_params = {}
        vit_params['embed_dim'] = 128
        vit_params['depth'] = 4
        vit_params['num_heads'] = 4
        vit_params['mlp_ratio'] = 3.0
        vit_params['hid_dim'] = 64
        vit_params['decoder_feat_HxW'] = 12*12
        from model.decoder.transformer_decoder import transformer_decoder
        decoder = transformer_decoder(vit_params, c1_channels=128, c4_channels=128)
        # need to fix the uniform API

    return decoder
