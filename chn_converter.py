def int_to_chinese(num):
    uni_str = '\\u' + hex(num)[2:]
    return uni_str.encode().decode()
