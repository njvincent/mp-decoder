import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import argparse 
import h5py
from matplotlib import cm, colors 

# ───────────────────────────────────────────────────────────────
def visualise(
        hist, minfields, *,
        interval=500,
        save_path=None, fps=4,
):
    """
    hist       : Bool   (T, Lx, Ly, Lz)
    minfields  : Float  (T, Lx, Ly, Lz)
    """
    T, Lx, Ly, Lz = hist.shape
    assert minfields.shape == (T, Lx, Ly, Lz)

    # ── pre-compute plane coordinates (xy grid at z = 0) ─────────
    y_idx, x_idx = np.mgrid[0:Lx, 0:Ly]   # y = i, x = j
    Z_plane = np.zeros_like(x_idx)        # same shape, all zeros

    # global colour limits for the heat-map
    sums = minfields.sum(axis=3)          # (T, Lx, Ly)
    vmin, vmax = sums.min(), sums.max()
    cmap = cm.get_cmap("Purples")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)     # linear 0‒1 scaler
    # face_colours0 = cmap(norm(sums[0]))
    face_colours0 = cmap(norm(sums[0][:-1, :-1]))
    # print(face_colours0)

    # ── figure, axes, slider layout ──────────────────────────────
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(7, 1)
    ax = fig.add_subplot(gs[:6, 0], projection="3d",computed_zorder=False)
    ax.view_init(elev=10., azim=45)
    ax_slider = fig.add_subplot(gs[6, 0])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_xlim(0, Ly - 1)
    ax.set_ylim(Lx - 1, 0)
    ax.set_zlim(0, Lz - 1)

    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))   # transparent panes
        axis._axinfo["grid"]["linewidth"] = 0
        axis.set_tick_params(which="both", length=0)
    ax.minorticks_off() 

    # ── draw the heat-map plane for frame 0 ──────────────────────
    surf = ax.plot_surface(
        x_idx, y_idx, Z_plane,
        rstride=1, cstride=1,
        facecolors=face_colours0,
        shade=False,
        antialiased=False,
        linewidth=0,
        zorder=1
    )

    # ── red spheres for hist frame 0 ─────────────────────────────
    ii, jj, kk = np.nonzero(hist[0])
    spheres = ax.scatter(
        jj, ii, kk,
        s=150, c="lime", marker="o",
        ec="k", lw=2,
        depthshade=not True,zorder=5
    )

    # ── slider widget ────────────────────────────────────────────
    slider = Slider(ax_slider, "t", 0, T - 1,
                    valinit=0, valstep=1, initcolor="none")

    # update routine shared by slider & animation
    def _draw_frame(t):
        nonlocal spheres 
        t = int(t)

        # 1) update plane colours
        new_fc = cmap(norm(sums[t][:-1, :-1])).reshape(-1, 4)   # flatten to (Nfaces, 4)
        surf.set_facecolors(new_fc)

        # 2) update red spheres
        spheres.remove()
        ii, jj, kk = np.nonzero(hist[t])
        spheres = ax.scatter(
            jj, ii, kk,
            s=150, c="lime", marker="o",
            ec="k", lw=2,
            depthshade=True,zorder=100
        )
        # spheres._offsets3d = (jj, ii, kk+1)

        ax.set_title(f"t = {t}")
        slider.eventson = False
        slider.set_val(t)
        slider.eventson = True
        return surf, spheres

    # slider callback
    slider.on_changed(lambda val: (_draw_frame(val), fig.canvas.draw_idle()))

    # arrow keys
    def _on_key(event):
        if event.key in ("left", "right"):
            step = -1 if event.key == "left" else 1
            new_t = (int(slider.val) + step) % T
            _draw_frame(new_t)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # ── build FuncAnimation (paused by default) ──────────────────
    # ani = FuncAnimation(fig, _draw_frame, frames=T,
    #                     interval=interval, blit=False, repeat=True)
    # ani.event_source.stop()            # ← no automatic play

    # # optional save
    # if save_path is not None:
    #     print(f"Saving animation → {save_path} ...")
    #     ani.save(save_path, fps=fps, dpi=200)
    #     print("done.")


    ani = None
    if save_path is not None:
        ani = FuncAnimation(fig, _draw_frame,
                            frames=T, blit=False, repeat=False)
        ani.save(save_path, fps=fps, dpi=200)

    plt.tight_layout()
    plt.show()
    return ani


# ───────────────────────── demo with dummy data ─────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-fin',default='no') # spacetime history of fields and spins 
    parser.add_argument('-save_animation',default='no') # saves animation for max_animation_time (defined below) if not equal to "no" 
    parser.add_argument('-state_hist',action='store_true',default=False) # if true, shows an image of the state history 
    args = parser.parse_args() 

    def loaddata(fin):
        with h5py.File(fin, "r") as f:
            key_list = [key for key in f.keys() if not key.startswith("_")] 
            data_dict = dict.fromkeys(key_list)
            for key in key_list:
                data_dict[key] = f[key][()]  
        return data_dict 

    # load field history data 
    hist_data = loaddata(args.fin)
    hist_raw = hist_data["hist"]
    if getattr(hist_raw.dtype, "fields", None) is not None:
        raise TypeError(
            "hist is stored as a Julia/JLD2 BitArray wrapper, not a plain array. "
            "Regenerate the demo after the CNOT_DEMO writer change, which saves hist as UInt8."
        )
    hist = hist_raw.T.astype(bool)
    print(np.shape(hist),np.sum(hist))
    L = hist_data["L"]
    Z = hist_data["Z"]
    T = np.shape(hist)[0]
    field_hist = hist_data["field_hist"].T
    field_hist[field_hist == 0] = 100000
    minfields = (1/np.min(field_hist[:,:,:,:,:,:],axis=(4,5)))**.75 # value of component(s) which controls motion 

    visualise(hist, minfields, interval=400, save_path=args.save_animation if args.save_animation != "no" else None, fps=4)
