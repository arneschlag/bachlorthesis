# Copyright (C) Robert Bosch GmbH 2018.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# This program and the accompanying materials are made available under
# the terms of the Bosch Internal Open Source License v4
# which accompanies this distribution, and is available at
# http://bios.intranet.bosch.com/bioslv4.txt


################################## ATTENTION #############################
# This file must be the same under "all packages/bolt-*" directories.    #
# Edits must only be done to "packages/bolt-core/build_backend.py".      #
# Any changes are propagated to the other packages by a pre-commit hook. #
##########################################################################


from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from wheel.wheelfile import WheelFile  # pants: no-infer-dep


def _run_pants(cwd, *extra_args):
    return subprocess.run(
        ["./pants", "--no-dynamic-ui", *extra_args, "package", f"packages/{cwd.name}:{cwd.name}"],
        cwd="../..",
        capture_output=True,
        stdin=subprocess.DEVNULL,
        check=True,
        encoding="utf-8",
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    """PEP-517 compatible hook for creation of wheel file.

    This function calls pants to create the wheel file and then provides it to the caller.
    """
    # Make sure we're running in the right directory (.../packages/bolt-*)
    cwd = Path.cwd()
    if cwd.parent.name != "packages" or not cwd.name.startswith("bolt-"):
        raise RuntimeError(f"Expected to run in a directory '.../packages/bolt-*', got {cwd=}")

    # Call pants to create the wheel file, retrying without pantsd when failing the first time
    try:
        ret = _run_pants(cwd)
    except subprocess.CalledProcessError:
        ret = _run_pants(cwd, "--no-watch-filesystem", "--no-pantsd")

    # Get the name of the wheel from the output of pants
    matches = re.search(r"Wrote (dist/.*\.whl)$", ret.stderr)
    if not matches:
        raise RuntimeError(f"Could not locate written file in the output of pants\n{ret.stdout=}\n{ret.stderr=}")
    output_file = Path("../..") / matches[1]

    # Copy the wheel to the wheel_directory and return its name
    shutil.copy(output_file, wheel_directory)
    return output_file.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    """PEP-517 compatible hook for creation of editable wheel file.

    This function uses :func:`build_wheel` to create a wheel file containing the source files and metadata. From this,
    it only takes the metadata, adds a ``.pth`` file and builds the editable wheel.

    Limitations:

        - It is assumed that the sources reside directly under the root of the package, not e.g. under sources/
    """
    output_wheel_name = build_wheel(wheel_directory, config_settings, metadata_directory)
    output_wheel_path = Path(wheel_directory) / output_wheel_name
    package_name = output_wheel_name.split("-")[0]

    with tempfile.TemporaryDirectory() as unpack_dir, tempfile.TemporaryDirectory() as clean_dir:
        # Unpack and delete the wheel file
        with WheelFile(output_wheel_path) as wf:
            wf.extractall(unpack_dir)
        output_wheel_path.unlink()

        # Move the *.dist-info folder to the clean directory
        print(list(Path(unpack_dir).iterdir()))
        dist_info_dirs = list(Path(unpack_dir).glob("*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise RuntimeError(f"Expected one .dist-info directory in wheel, got {dist_info_dirs=}")
        shutil.move(str(dist_info_dirs[0]), clean_dir)

        # Write the .pth file
        (Path(clean_dir) / f"{package_name}.pth").write_text(f"{Path.cwd()}\n")

        # Re-pack the wheel into the same file as before
        with WheelFile(output_wheel_path, "w") as wf:
            wf.write_files(clean_dir)

    return output_wheel_name
