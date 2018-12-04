# header names
from enum import Enum

HEADER_STRING_TIMESTAMP = 'Timestamp'
HEADER_STRING_NORMAL_OR_ATTACK = 'Normal/Attack'


class SrcType(Enum):
    SENSOR = 0
    ACTUATOR = 1


class DataType(Enum):
    ANALOG = 0
    DIGITAL = 1


SOURCES = [
    ('MV101', 1, SrcType.ACTUATOR, DataType.DIGITAL),  # motorized valve; controls water flow to the raw water tank
    ('P101', 1, SrcType.ACTUATOR, DataType.DIGITAL),  # pump; pumps water from raw water tank to second stage
    ('P102', 1, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-101)
    ('FIT101', 1, SrcType.SENSOR, DataType.ANALOG),  # flow meter; measures inflow into raw water tank
    ('LIT101', 1, SrcType.SENSOR, DataType.ANALOG),  # level transmitter; raw water tank level
    ('MV201', 2, SrcType.ACTUATOR, DataType.DIGITAL),
    # motorized valve; controls water flow to the UF feed water tank
    ('P201', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # Dosing pump; NaCl dosing pump
    ('P202', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-201)
    ('P203', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # Dosing pump; HCl dosing pump
    ('P204', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-203)
    ('P205', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # Dosing pump; NaOCl dosing pump
    ('P206', 2, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-205)
    ('AIT201', 2, SrcType.SENSOR, DataType.ANALOG),  # conductivity analyzer; measures NaCl level
    ('AIT202', 2, SrcType.SENSOR, DataType.ANALOG),  # pH analyzer; measure HCl level
    ('AIT203', 2, SrcType.SENSOR, DataType.ANALOG),  # ORP analyzer; measures NaOCl level
    ('FIT201', 2, SrcType.SENSOR, DataType.ANALOG),  # flow transmitter; control dosing pumps
    ('MV301', 3, SrcType.ACTUATOR, DataType.DIGITAL),  # motorized value; controls UF-Backwash process
    ('MV302', 3, SrcType.ACTUATOR, DataType.DIGITAL),
    # motorized value; controls water from UF process to De-Chlorination unit
    ('MV303', 3, SrcType.ACTUATOR, DataType.DIGITAL),  # motorized value; controls UF-Backwash drain
    ('MV304', 3, SrcType.ACTUATOR, DataType.DIGITAL),  # motorized value; controls UF drain
    ('P301', 3, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-302)
    ('P302', 3, SrcType.ACTUATOR, DataType.DIGITAL),
    # UF feed pump; pumps water from UF feed water tank to RO feed water tank via UF filtration
    ('DPIT301', 3, SrcType.SENSOR, DataType.ANALOG),
    # differential pressure indicating transmitter; controls the backwash process
    ('FIT301', 3, SrcType.SENSOR, DataType.ANALOG),  # flow meter; measure the flow of water in the UF stage
    ('LIT301', 3, SrcType.SENSOR, DataType.ANALOG),  # level transmitter; UF feed water tank level
    ('P401', 4, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-402)
    ('P402', 4, SrcType.ACTUATOR, DataType.DIGITAL),  # pumps; pumps water from RO feed tank to UV dechlorinator
    ('P403', 4, SrcType.ACTUATOR, DataType.DIGITAL),  # sodium bi-sulphate pump
    ('P404', 4, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-403)
    ('UV401', 4, SrcType.ACTUATOR, DataType.DIGITAL),  # dechlorinator; removes chlorine from water
    ('AIT401', 4, SrcType.SENSOR, DataType.ANALOG),  # RO hardness meter of water
    ('AIT402', 4, SrcType.SENSOR, DataType.ANALOG),
    # ORP meter; controls the NaHSO3 dosing(P203), NaOCl dosing (P205)
    ('FIT401', 4, SrcType.SENSOR, DataType.ANALOG),  # flow transmitter; controls the UV dechlorinator
    ('LIT401', 4, SrcType.ACTUATOR, DataType.ANALOG),  # level transmitter; RO feed water tank level
    ('P501', 5, SrcType.ACTUATOR, DataType.DIGITAL),  # pump; pumps dechlorinated water to RO
    ('P502', 5, SrcType.ACTUATOR, DataType.DIGITAL),  # backup (P-501)
    ('AIT501', 5, SrcType.SENSOR, DataType.ANALOG),  # RO pH analyzer; measures HCl level
    ('AIT502', 5, SrcType.SENSOR, DataType.ANALOG),  # RO feed ORP analyzer; measures NaOCl level
    ('AIT503', 5, SrcType.SENSOR, DataType.ANALOG),  # RO feed conductivity analyzer; measures NaCl level
    ('AIT504', 5, SrcType.SENSOR, DataType.ANALOG),  # RO permeate conductivity analyzer; measures NaCl level
    ('FIT501', 5, SrcType.SENSOR, DataType.ANALOG),  # flow meter; RO membrane inlet flow meter
    ('FIT502', 5, SrcType.SENSOR, DataType.ANALOG),  # flow meter; RO permeate flow meter
    ('FIT503', 5, SrcType.SENSOR, DataType.ANALOG),  # flow meter; RO reject flow meter
    ('FIT504', 5, SrcType.SENSOR, DataType.ANALOG),  # flow meter; RO re-circulation flow meter
    ('PIT501', 5, SrcType.SENSOR, DataType.ANALOG),  # pressure meter; RO feed pressure
    ('PIT502', 5, SrcType.SENSOR, DataType.ANALOG),  # pressure meter; RO permeate pressure
    ('PIT503', 5, SrcType.SENSOR, DataType.ANALOG),  # pressure meter; RO reject pressure
    #   'P601' [actuator] pump; pumps water from RO permeate tank to raw water tank (not used for data collection)
    ('P602', 6, SrcType.ACTUATOR, DataType.DIGITAL),
    # pump; pumps water from UF back wash tank to UF filter to clean the membrane
    #   'P603', [actuator] not implemented in SWaT yet
    ('FIT601', 6, SrcType.SENSOR, DataType.ANALOG),  # flow meter; UF backwash flow meter
]

ANALOG_SRCS = [src[0] for src in SOURCES if src[3] == DataType.ANALOG]
DIGITAL_SRCS = [src[0] for src in SOURCES if src[3] == DataType.DIGITAL]
ALL_SRCS = ANALOG_SRCS + DIGITAL_SRCS


def srcs_in_process(idx):
    return [src[0] for src in SOURCES if src[1] == idx]


def index_in_process(pidx):
    return [idx for idx, src in enumerate(SOURCES) if src[1] == pidx]


N_PROCESS = 6
P_SRCS = [srcs_in_process(pidx + 1) for pidx in range(N_PROCESS)]
P_IDXS = [index_in_process(pidx + 1) for pidx in range(N_PROCESS)]

# number of sources
N_DIGITAL_SRCS = len(DIGITAL_SRCS)
N_ANALOG_SRCS = len(ANALOG_SRCS)
N_ALL_SRCS = len(SOURCES)

# dataset
WINDOW_SIZE = 100
WINDOW_GIVEN = 90
WINDOW_PREDICT = 9

SWaT_TIME_FORMAT = '%d/%m/%Y %I:%M:%S %p'
INFLUX_RETURN_TIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
DATETIME_BASIC_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

EVAL_MEASUREMENT = 'seq2seq_eval_teach_p{}'
TRAIN_LOSS_MEASUREMENT = 'seq2seq_train_loss_p{}'
JUDGE_MEASUREMENT = 'seq2seq_judge_p{}'

ANOMALY_PROBS_PICKLE = 'dat/digged_probs.dat'

N_HIDDEN_CELLS = 64
EVALUATION_NORM_P = 4
