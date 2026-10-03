"""An omni.ui slider panel for driving the robot by hand, inside the daemon.

Why in the daemon rather than a separate script: the daemon already owns the
stage, the articulation and the step loop, and serve() applies `daemon.unit`
on every step while playing. A slider only has to write into that same
vector and the robot follows on the next frame -- no second
SimulationContext (it is a singleton) and no polling over the socket.

The socket stays live alongside this. Sliders write the shared target; a
set_targets from MCP overrides it, but the slider widgets do not move to
reflect it.

Values are the same normalized 0..1 the socket uses: 0 = lower joint limit,
1 = upper.
"""

from typing import Any

# Deliberately small. This exists to answer "what does this pose look like",
# and every additional control is one more thing to keep in sync with the
# socket's view of the same state.
_WIDTH = 420
_HEIGHT = 340


def build_slider_window(daemon: Any) -> Any:
    """Create the panel. Returns the window, or None if it cannot be built.

    Never raises. The panel is a convenience; the daemon and its socket are
    the product, and a UI that fails to build must not take them down. The
    docstring said this before the implementation did -- the body raised
    straight out to serve() and killed the daemon on load.
    """
    try:
        return _build(daemon)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[sliders] panel NOT built: {type(exc).__name__}: {exc}")
        print("[sliders] daemon and socket are unaffected; drive it over MCP")
        return None


def _build(daemon: Any) -> Any:
    try:
        import omni.ui as ui
    except ImportError:
        print("[sliders] omni.ui unavailable (headless?); no panel built")
        return None

    import torch

    scene = daemon.scene
    if scene is None:
        print("[sliders] no scene yet; no panel built")
        return None

    names = list(scene.driven_names)
    if daemon.unit is None:
        daemon.unit = torch.zeros(
            scene.num_driven, dtype=torch.float32, device=scene.sim.device)

    # Kit drops widgets whose Python references are garbage collected, so the
    # models have to outlive this function. They are parked on the daemon at
    # the end of this function.
    models = []

    def _set(index: int, value: float) -> None:
        daemon.unit[index] = float(value)

    def _set_all(value: float) -> None:
        for i, m in enumerate(models):
            m.set_value(value)      # fires the value_changed callback
            _set(i, value)

    window = ui.Window(daemon.robot.name, width=_WIDTH, height=_HEIGHT)
    # Scrolls, so a robot with many driven joints still fits.
    with window.frame, ui.ScrollingFrame(), ui.VStack(spacing=6, height=0):
        ui.Label("Driven joints — 0 = lower limit, 1 = upper",
                 height=20, style={"font_size": 14})

        for i, name in enumerate(names):
            with ui.HStack(height=24, spacing=6):
                ui.Label(name, width=150)
                # No step= kwarg: FloatSlider does not take one, and
                # passing it raises inside Kit's binding layer.
                slider = ui.FloatSlider(min=0.0, max=1.0)
                slider.model.set_value(float(daemon.unit[i].item()))
                slider.model.add_value_changed_fn(
                    lambda m, idx=i: _set(idx, m.get_value_as_float()))
                models.append(slider.model)

        ui.Spacer(height=8)
        with ui.HStack(height=28, spacing=6):
            ui.Button("All lower", clicked_fn=lambda: _set_all(0.0))
            ui.Button("All upper", clicked_fn=lambda: _set_all(1.0))

        with ui.HStack(height=28, spacing=6):
            # The daemon starts paused so a script can set everything up
            # before time advances. A slider panel is useless paused, so
            # these are the first thing most sessions reach for.
            ui.Button("Play", clicked_fn=lambda: setattr(daemon, "playing", True))
            ui.Button("Pause", clicked_fn=lambda: setattr(daemon, "playing", False))


    # Keep-alive on the DAEMON, not the window: omni.ui objects are pybind11
    # wrappers with no __dict__, so setting an attribute on one raises
    # AttributeError and takes the panel down on load.
    daemon._slider_models = models
    print(f"[sliders] panel up for {len(names)} driven joints")
    return window
