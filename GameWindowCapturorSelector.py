import platform

if platform.system() == 'Darwin':
    from GameWindowCapturorForMacbook import GameWindowCapturor
else:
    from GameWindowCapturor import GameWindowCapturor 