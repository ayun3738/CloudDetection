from roboflow import Roboflow

# cloud_640s
# rf = Roboflow(api_key="odXiNgbNhpl15seknp9L")
# project = rf.workspace("alpaco-od").project("cloud_classification-7edfv")
# dataset = project.version(14).download("yolov8")

# cloud_ASV
# rf = Roboflow(api_key="3rEX1pUJj5lgWa96L97x")
# project = rf.workspace("cloudasv").project("cloud_classification_asv")
# dataset = project.version(1).download("yolov8")

# cloud_3class
# rf = Roboflow(api_key="HBnD9ps6unpkOsXhTh7w")
# project = rf.workspace("cloudsuvel").project("cloud_classification_suvel")
# dataset = project.version(1).download("yolov8")

# cloud_ASV2
rf = Roboflow(api_key="Wu6qzSJUlQbvCH4MBvxt")
project = rf.workspace("cloudasv2").project("cloud_classification_asv-5ggrb")
dataset = project.version(1).download("yolov8")
