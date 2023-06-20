from Core.Saxs2dProfile import Saxs2dProfile
import PySimpleGUI as sg
import numpy as np

from .. import util
from .FigCanvas import FlushableFigCanvas

logger = util.getLogger(__name__, util.DEBUG)

TMPDIR = "tmp/"

WINDOW_SIZE = (500, 750)
CANVAS_SIZE = (500, 600)

STATE_INIT = "init"
STATE_WAIT_AUTO_MASK = "wait_auto_mask"
STATE_WAIT_DETECT_CENTER = "wait_detect_center"
STATE_WAIT_SELECT_CENTER = "wait_select_center"
STATE_WAIT_INTEGRATE = "wait_integrate"


def main():
    layout = [
        [
            sg.Text(f"path:"),
            sg.InputText(
                default_text="testdata/s194sUn/s194sUn00000_00480.tif",
                key="-INPUT_FILEPATH-",
            ),
        ],
        [sg.Button("-", key="-BUTTON_ACTION-")],
        [
            sg.Text("status:", key="-TEXT_STATUS_HEADER-"),
            sg.Text("successfully initiated", key="-TEXT_STATUS-"),
        ],
        [sg.Canvas(size=CANVAS_SIZE, key="-CANVAS-")],
        [
            sg.Button("exit", key="-BUTTON_EXIT-"),
            sg.Button("save", key="-BUTTON_SAVE-"),
        ],
    ]

    window = sg.Window("test", layout, size=WINDOW_SIZE, finalize=True)

    action_button = window["-BUTTON_ACTION-"]
    status = window["-TEXT_STATUS-"]
    update_status = lambda mes: status.update(value=mes)

    figCanvas = FlushableFigCanvas(window["-CANVAS-"].TKCanvas)  # type: ignore

    state = STATE_INIT
    action_button.update(text="load")
    profile: Saxs2dProfile = None  # type: ignore
    while True:
        event, values = window.read(timeout=100)  # type: ignore
        if event != sg.TIMEOUT_KEY:
            logger.debug(f"state:{state} event: {event}")

        if event == "-BUTTON_EXIT-" or event == sg.WIN_CLOSED:
            break

        if event == "-BUTTON_SAVE-":
            savefile = TMPDIR + "test.png"
            profile.save(savefile, overwrite=True, showCenter=True)
            update_status(f"saved to `{savefile}`")

        if state == STATE_INIT and event == "-BUTTON_ACTION-":
            filepath = values["-INPUT_FILEPATH-"]
            try:
                profile = Saxs2dProfile.load_tiff(filepath)
            except FileNotFoundError:
                window["-TEXT_STATUS-"].update(value="file not found")
                update_status("file not found")
                continue
            except ValueError:
                window["-TEXT_STATUS-"].update(value="invalid file type")
                update_status("invalid file type")
                continue

            figCanvas.heatmap(profile.values(showMaskAsNan=False))
            update_status(f"`{filepath}` successfully loaded")
            state = STATE_WAIT_AUTO_MASK
            action_button.update(text="auto mask")
            continue

        if state == STATE_WAIT_AUTO_MASK and event == "-BUTTON_ACTION-":
            profile.auto_mask_invalid()
            figCanvas.heatmap(profile.values())
            update_status("auto mask done")
            state = STATE_WAIT_DETECT_CENTER
            action_button.update(text="detect center")
            continue
        if state == STATE_WAIT_DETECT_CENTER and event == "-BUTTON_ACTION-":
            profile.detect_center()
            if profile.center is None:
                update_status("center not detected.\twaiting for manual selection")
                state = STATE_WAIT_SELECT_CENTER
                continue
            else:
                figCanvas.heatmap(profile.values(showCenterAsNan=True))
                update_status("center detected")
                action_button.update(text="integrate")
                state = STATE_WAIT_INTEGRATE
                continue

        if state == STATE_WAIT_SELECT_CENTER and event == "-BUTTON_ACTION-":
            update_status("center selection called but not implemented yet")
            continue

        if state == STATE_WAIT_INTEGRATE and event == "-BUTTON_ACTION-":
            y, bins = profile.integrate(dr=5.0)
            x = (bins[:-1] + bins[1:]) / 2
            y = np.log(y)
            logger.debug(f"integrated: {x.shape}, {y.shape}")
            figCanvas.plot(x, y)
            update_status("integrated")
            continue

    window.close()


if __name__ == "__main__":
    main()
