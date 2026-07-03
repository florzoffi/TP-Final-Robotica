import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/isidro_valeriano/ws/src/TP-Final-Robotica/install/tpf'
