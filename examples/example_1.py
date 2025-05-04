"""Example 1."""

import logging
import pathlib

import numpy as np

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

known_properties = {
    "PUDXES": {
        "centre_of_mass": np.array([12.4, 12.4, 12.4]),
        "maximum_diameter": {
            "atom_1": 12,
            "atom_2": 54,
            "diameter": 22.179369990077188,
        },
        "no_of_atoms": 168,
        "pore_diameter": {"atom": 29, "diameter": 5.3970201773100221},
        "pore_diameter_opt": {
            "atom_1": 29,
            "centre_of_mass": np.array([12.4, 12.4, 12.4]),
            "diameter": 5.3970201773100221,
        },
        "pore_volume": 82.311543851544172,
        "pore_volume_opt": 82.311543851544172,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [10.77105705, 10.77097707, 14.02893956],
                    [14.01544846, 14.0154126, 14.01539845],
                    [13.92965524, 10.87029766, 10.87034163],
                    [10.77542236, 14.02453217, 10.77546634],
                ]
            ),
            "diameters": np.array(
                [3.63778746, 3.63562103, 3.62896512, 3.63707237]
            ),
        },
        "average_diameter": 13.83201751425547,
    },
    "YAQHOQ": {
        "centre_of_mass": np.array(
            [-1.66666667e-05, -4.92980557e-18, 3.33333333e-05]
        ),
        "maximum_diameter": {
            "atom_1": 43,
            "atom_2": 54,
            "diameter": 10.495187523948891,
        },
        "no_of_atoms": 60,
        "pore_diameter": {"atom": 11, "diameter": 3.6101494139251806},
        "pore_diameter_opt": {
            "atom_1": 11,
            "centre_of_mass": np.array([0.00332048, 0.01618053, -0.00052978]),
            "diameter": 3.6289842522285096,
        },
        "pore_volume": 24.636224433953796,
        "pore_volume_opt": 25.023835308827408,
        "windows": {"centre_of_mass": None, "diameters": None},
        "average_diameter": 10.016651194000373,
    },
    "BATVUP": {
        "centre_of_mass": np.array([9.78711345, 4.8907307, 10.42542589]),
        "maximum_diameter": {
            "atom_1": 32,
            "atom_2": 84,
            "diameter": 14.779624994419478,
        },
        "no_of_atoms": 108,
        "pore_diameter": {"atom": 106, "diameter": 4.8365345185135187},
        "pore_diameter_opt": {
            "atom_1": 106,
            "centre_of_mass": np.array([9.7871141, 4.87112257, 10.47998608]),
            "diameter": 4.952487834793544,
        },
        "pore_volume": 59.23815140453344,
        "pore_volume_opt": 63.601721675049134,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [9.50377026, 7.1542064, 12.84136045],
                    [9.25703433, 2.72286828, 8.13923095],
                ]
            ),
            "diameters": np.array([3.72937988, 3.34146021]),
        },
        "average_diameter": 12.323749433141906,
    },
    "NUXHIZ": {
        "centre_of_mass": np.array([8.54082679, 11.35269286, 19.7892421]),
        "maximum_diameter": {
            "atom_1": 43,
            "atom_2": 57,
            "diameter": 18.586437224893469,
        },
        "no_of_atoms": 138,
        "pore_diameter": {"atom": 17, "diameter": 8.7465283605673907},
        "pore_diameter_opt": {
            "atom_1": 17,
            "centre_of_mass": np.array([8.32795441, 11.51683898, 19.6255358]),
            "diameter": 8.968163467457108,
        },
        "pore_volume": 350.35292555652876,
        "pore_volume_opt": 377.6671140718386,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [11.83772373, 11.26384234, 22.41304403],
                    [6.14465038, 15.40381028, 19.5164307],
                    [6.80395224, 6.7378818, 17.72154653],
                ]
            ),
            "diameters": np.array([6.5036498, 7.90390212, 7.26955977]),
        },
        "average_diameter": 16.65433095190861,
    },
    "REYMAL": {
        "centre_of_mass": np.array(
            [1.28912864e01, 1.28912864e01, 2.47150281e-16]
        ),
        "maximum_diameter": {
            "atom_1": 75,
            "atom_2": 192,
            "diameter": 33.957689421982842,
        },
        "no_of_atoms": 468,
        "pore_diameter": {"atom": 280, "diameter": 13.756212487538123},
        "pore_diameter_opt": {
            "atom_1": 397,
            "centre_of_mass": np.array(
                [1.28908997e01, 1.28912452e01, 5.00595384e-06]
            ),
            "diameter": 13.756740717062886,
        },
        "pore_volume": 1362.9980958535011,
        "pore_volume_opt": 1363.155116577926,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [13.00073477, 12.76534416, 6.39425694],
                    [18.14169824, 16.36587283, 0.13593832],
                    [16.38477402, 7.61303639, -0.13629682],
                    [7.75757301, 9.49388882, 0.13395727],
                    [9.31073034, 18.29908143, -0.13827555],
                    [13.01407718, 12.99765472, -6.30127025],
                ]
            ),
            "diameters": np.array(
                [
                    9.05947034,
                    9.17248674,
                    9.17507083,
                    9.16546626,
                    9.1922052,
                    9.05410173,
                ]
            ),
        },
        "average_diameter": 25.261360227143175,
    },
}


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"
    output_directory = data_directory / "output"

    input_files = [
        input_directory / "PUDXES.xyz",
        input_directory / "YAQHOQ.xyz",
        input_directory / "BATVUP.xyz",
        input_directory / "NUXHIZ.xyz",
        input_directory / "REYMAL.xyz",
    ]
    for input_file in input_files:
        name = input_file.name.split(".")[0]

        molsys = pw.MolecularSystem.load_file(input_file)
        # If no preprocessing of the structure is required we can pass it
        # directly to the Molecule class using
        # MolecularSystem.system_to_molecule method.
        mol = molsys.system_to_molecule()

        # Individual analysis.
        mol.calculate_centre_of_mass()
        mol.calculate_maximum_diameter()
        mol.calculate_average_diameter()
        mol.calculate_pore_diameter()
        mol.calculate_pore_volume()
        mol.calculate_pore_diameter_opt()
        mol.calculate_pore_volume_opt()
        mol.calculate_windows()

        mol.full_analysis()
        logger.info("properties for %s: %s", input_file.name, mol.properties)
        (same_dict, failed_prop) = pw.compare_properties_dict(
            dict1=mol.properties,  # type:ignore[arg-type]
            dict2=known_properties[name],  # type:ignore[arg-type]
        )

        if not same_dict:
            msg = (
                f"mol.properties not the same as known for "
                f"{input_file.name} in property: {failed_prop}"
                f"\n {mol.properties}"
            )
            raise RuntimeError(msg)

        mol.dump_properties_json(
            filepath=output_directory / f"{name}_out.json",
            override=True,
        )

        mol.dump_molecule(
            filepath=output_directory / f"{name}_out.pdb",
            include_coms=True,
            override=True,
        )


if __name__ == "__main__":
    main()
