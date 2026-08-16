from tqdm import tqdm
from prepare import prepare_project
from process import process_project
from score import score_project

datasets = {
    "mvtec_ad": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
}

if __name__ == "__main__":
    for source, projects in datasets.items():
        for project in projects:
            try:
                prepare_project(source, project)
                process_project(project, force_from="crop")
                score_project(project)

            except Exception as e:
                print(f"{source}: {project} FAILED")
                print(e)
