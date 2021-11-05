def get_neck(option, in_channel_list):
    if option['neck'].lower() == 'aspp':
        from model.neck.neck import aspp_neck
        neck = aspp_neck(in_channel_list, option['neck_channel'])
    elif option['neck'].lower() == 'basic':
        from model.neck.neck import basic_neck
        neck = basic_neck(in_channel_list, option['neck_channel'])

    return neck
