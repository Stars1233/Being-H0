
from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset
from BeingH.dataset.datasets.vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    'libero_posttrain': LeRobotIterableDataset,
    'robocasa_human_posttrain': LeRobotIterableDataset,
    'uni_posttrain': LeRobotIterableDataset,
}


DATASET_INFO = {
    'libero_posttrain': {
        'libero_spatial': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        },
        'libero_object': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        },
        'libero_goal': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        },
        'libero_10': {
            'dataset_path': "/share/dataset/beingh_posttrain/libero/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        },
    },

    'robocasa_human_posttrain': {
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseDoubleDoor",
        },
        'single_panda_gripper.CloseDrawer': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseDrawer",
        },
        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CloseSingleDoor",
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeePressButton",
        },
        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeeServeMug",
        },
        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/CoffeeSetupMug",
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenDoubleDoor",
        },
        'single_panda_gripper.OpenDrawer': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenDrawer",
        },
        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/OpenSingleDoor",
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCabToCounter",
        },
        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToCab",
        },
        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToMicrowave",
        },
        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToSink",
        },
        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPCounterToStove",
        },
        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPMicrowaveToCounter",
        },
        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPSinkToCounter",
        },
        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/PnPStoveToCounter",
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffMicrowave",
        },
        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffSinkFaucet",
        },
        'single_panda_gripper.TurnOffStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOffStove",
        },
        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnMicrowave",
        },
        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnSinkFaucet",
        },
        'single_panda_gripper.TurnOnStove': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnStove",
        },
        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnSinkSpout",
        },
    },

    'uni_posttrain': {
        # ========================================================================
        # ROBOCASA datasets
        # ========================================================================
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseDoubleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDoubleDoor',
        },

        'single_panda_gripper.CloseDrawer': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseDrawer",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDrawer',
        },

        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CloseSingleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseSingleDoor',
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeePressButton",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeePressButton',
        },

        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeeServeMug",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeServeMug',
        },

        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/CoffeeSetupMug",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeSetupMug',
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenDoubleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDoubleDoor',
        },

        'single_panda_gripper.OpenDrawer': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenDrawer",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDrawer',
        },

        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/OpenSingleDoor",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenSingleDoor',
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCabToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCabToCounter',
        },

        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToCab",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToCab',
        },

        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToMicrowave',
        },

        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToSink",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToSink',
        },

        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPCounterToStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToStove',
        },

        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPMicrowaveToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPMicrowaveToCounter',
        },

        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPSinkToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPSinkToCounter',
        },

        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/PnPStoveToCounter",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPStoveToCounter',
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffMicrowave',
        },

        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffSinkFaucet",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffSinkFaucet',
        },

        'single_panda_gripper.TurnOffStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOffStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffStove',
        },

        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': "/share/dataset/beingh_posttrain/robocasa_human/single_stage/TurnOnMicrowave",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnMicrowave',
        },

        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOnSinkFaucet",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnSinkFaucet',
        },

        'single_panda_gripper.TurnOnStove': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnOnStove",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnStove',
        },

        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': "/share/dataset/beingh_real/posttrain/ROBOCASA/TurnSinkSpout",
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnSinkSpout',
        },
    },  
}