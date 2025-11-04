# Sleep-EDF (R&K) anotacijų mapinimas į 5-klasių AASM stilių:
# Originalios: W, 1, 2, 3, 4, R, M, ?
# Map: 3->N3, 4->N3; M ir ? – šaliname
LABEL_MAP = {
    "W": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",
    "R": "REM"
}
CLASSES = ["W", "N1", "N2", "N3", "REM"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in enumerate(CLASSES)}
