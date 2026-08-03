import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np


def interpolate_to_depth(ds, target_depths, depth_dim="levgrnd", var="TSOI"):
    sm = ds[var].squeeze()
    depths = ds[depth_dim].values.flatten()
    time = ds.indexes['time']
    
    values = []

    for x in target_depths:
        target_depth = x.values

        # Interpolate
        idx = int(np.searchsorted(depths, target_depth, side="right") - 1)
        idx = int(np.clip(idx, 0, len(depths) - 2))
        z0, z1 = depths[idx], depths[idx + 1]
        w = (target_depth - z0) / (z1 - z0)
        sm_interp = ((1 - w) * sm.values[:, idx] + w * sm.values[:, idx + 1]).flatten()
        values.append(sm_interp)

    interpolated_np = np.column_stack(values)


    interpolated_ds = xr.Dataset(
    {f"{var}": (["time", "depth"], interpolated_np, {"units": "K"})},
    coords={
        "time":  time,
        "depth": target_depths.values,
    },
    attrs={}
    )

    # Convert K → °C
    interpolated_ds = interpolated_ds - 273.15

    return interpolated_ds



def final_plot_tsoi(obs, sim, site, depths, vari, show=0):
    # Resample observational data to daily frequency
    obs_daily = obs.resample(time="1D").mean()

    for x in depths:

        plt.figure(figsize=(14,6))

        # Observed (daily)
        plt.plot(
            obs_daily["time"].values,
            obs_daily.sel(depth=x.values, method = "nearest").values,
            label="Observed",
            linewidth=2
        )

        # Simulated
        act_val = round(float(x.values),2)
        plt.plot(
            sim["time"].values,
            sim.sel(depth=act_val, method = "nearest")[f"{vari}"].values,
            label="Simulated",
            linewidth=2
        )

        # Labeling
        remain = int(round(x.values - act_val,3)*1000)
    
        plt.xlabel("Time")
        plt.ylabel(f"{vari}")
        plt.title(f"{vari} at {site} - {act_val*100:.0f}cm")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"v1.0_final_images/{site}/{vari}/{site}_{vari}_{act_val*100:.0f}cm_{remain}.png")
        if not show:
            plt.close()



def final_plot_h2osoi(obs, sim, site, depths, vari, show=0):
    # Resample observational data to daily frequency
    obs_daily = obs.resample(time="1D").mean()

    for x in depths:

        plt.figure(figsize=(14,6))

        # Observed (daily)
        act_val = round(float(x.values),2)
        plt.plot(
            obs_daily["time"].values,
            obs_daily.sel(depth=x.values, method = "nearest").values,
            label=f"Observed - {act_val*100:.0f}cm",
            linewidth=2
        )

        # Simulated
        
        sim_depth = sim.sel(levsoi=act_val, method = "nearest").levsoi.values
        
        plt.plot(
            sim["time"].values,
            sim.sel(levsoi=act_val, method = "nearest").values,
            label=f"Simulated - {sim_depth*100:.0f}cm",
            linewidth=2
        )

        # Labeling
        remain = int(round(x.values - act_val,3)*1000)
    
        plt.xlabel("Time")
        plt.ylabel(f"{vari}")
        plt.title(f"{vari} at {site}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"v1.0_final_images/{site}/{vari}/{site}_{vari}_{act_val*100:.0f}cm_{remain}.png")
        if not show:
            plt.close()


def final_plot_sd(obs, sim, site, depths, vari, show=0):
    # Resample observational data to daily frequency
    obs_daily = obs.resample(time="1D").mean()

    for x in depths:

        plt.figure(figsize=(14,6))

        # Observed (daily)
        plt.plot(
            obs_daily["time"].values,
            obs_daily.sel(depth=x.values, method = "nearest").values/1000,
            label=f"Observed",
            linewidth=2
        )

        # Simulated
        
        plt.plot(
            sim["time"].values,
            sim.values,
            label=f"Simulated",
            linewidth=2
        )

    
        plt.xlabel("Time")
        plt.ylabel(f"{vari}")
        plt.title(f"{vari} at {site} - {x.values:.0f}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"v1.0_final_images/{site}/{vari}/{site}_{vari}_{x.values}.png")
        if not show:
            plt.close()



def august_profile_tsoi(obs, sim, site, depths, vari, show=0):
    """
    Plot a soil temperature-vs-depth profile averaged over August,
    comparing observed and simulated values.
    """
    # Resample obs to daily (same as final_plot_tsoi)
    obs_daily = obs.resample(time="1D").mean()

    # Restrict to August
    obs_aug = obs_daily.sel(time=obs_daily["time"].dt.month == 8)
    sim_aug = sim.sel(time=sim["time"].dt.month == 8)

    # Convert K → °C
    sim_aug = sim_aug - 273.15

    depth_cm = []
    obs_vals = []
    sim_vals = []

    for x in depths:
        #print(x)
        act_val = round(float(x.values), 2)
        depth_cm.append(act_val * 100)

        obs_vals.append(
            obs_aug.sel(depth=x.values, method="nearest").mean(skipna=True).values
        )
        sim_vals.append(
            sim_aug.sel(levgrnd=act_val, method="nearest").mean(skipna=True).values
        )

    plt.figure(figsize=(6, 8))
    plt.plot(obs_vals, depth_cm, "o-", label="Observed", linewidth=2)
    plt.plot(sim_vals, depth_cm, "s-", label="Simulated", linewidth=2)

    plt.gca().invert_yaxis()  # depth increases downward
    plt.xlabel(f"{vari}")
    plt.ylabel("Depth (cm)")
    plt.title(f"{vari} Profile at {site} - August Average")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"v1.0_final_images/{site}/{vari}/{site}_{vari}_profile.png")
    if not show:
        plt.close()



def clean_august_profile_tsoi(obs, sim, site, depths, vari, show=0):
    """
    Plot a soil temperature-vs-depth profile averaged over August,
    comparing observed and simulated values.

    Returns
    -------
    dict
        Mapping of {act_val (rounded depth, m): raw_xval (unrounded depth, m)}
        for the depths actually used in the plot.
    """
    # Resample obs to daily (same as final_plot_tsoi)
    obs_daily = obs.resample(time="1D").mean()
    # Restrict to August
    obs_aug = obs_daily.sel(time=obs_daily["time"].dt.month == 8)
    sim_aug = sim.sel(time=sim["time"].dt.month == 8)
    # Convert K → °C
    sim_aug = sim_aug - 273.15

    profile_data = {}  # act_val -> (obs_val, sim_val, raw_xval)
    for x in depths:
        act_val = round(float(x.values), 2)
        obs_val = obs_aug.sel(depth=x.values, method="nearest").mean(skipna=True).values
        sim_val = sim_aug.sel(levgrnd=act_val, method="nearest").mean(skipna=True).values

        if act_val not in profile_data or abs(obs_val - sim_val) < abs(
            profile_data[act_val][0] - profile_data[act_val][1]
        ):
            profile_data[act_val] = (obs_val, sim_val, float(x.values))

    sorted_depths = sorted(profile_data)
    depth_cm = [d * 100 for d in sorted_depths]
    obs_vals = [profile_data[d][0] for d in sorted_depths]
    sim_vals = [profile_data[d][1] for d in sorted_depths]
    used_xvals = {d: profile_data[d][2] for d in sorted_depths}

    plt.figure(figsize=(6, 8))
    plt.plot(obs_vals, depth_cm, "o-", label="Observed", linewidth=2)
    plt.plot(sim_vals, depth_cm, "s-", label="Simulated", linewidth=2)
    plt.gca().invert_yaxis()  # depth increases downward
    plt.xlabel(f"{vari}")
    plt.ylabel("Depth (cm)")
    plt.title(f"{vari} Profile at {site} - August Average")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"v1.0_final_images/{site}/{vari}/clean_{site}_{vari}_profile.png")
    if not show:
        plt.close()

    return used_xvals


def seasonal_cycle_tsoi(obs, sim, site, used_xvals, vari, show=0):
    """
    Plot the seasonal (monthly) cycle of a depth-averaged variable,
    comparing observed and simulated values. Shading shows the
    min-max range across years for each calendar month.

    Parameters
    ----------
    obs, sim : xr.DataArray
        Observed (has 'depth' dim, m) and simulated (has 'levgrnd' dim, m)
        time series.
    site : str
    used_xvals : dict
        {act_val (rounded depth, m): raw_xval (unrounded obs depth, m)}
        as returned by august_profile_tsoi — the depths to average over.
    vari : str
    show : int
    """
    # Resample obs to daily
    obs_daily = obs.resample(time="1D").mean()
    # Convert sim K -> degC
    sim_c = sim - 273.15

    # Average over the relevant depths (same nearest-match logic as before)
    obs_depth_list = []
    sim_depth_list = []
    for act_val, raw_xval in used_xvals.items():
        obs_depth_list.append(obs_daily.sel(depth=raw_xval, method="nearest"))
        sim_depth_list.append(sim_c.sel(levgrnd=act_val, method="nearest"))

    obs_avg = xr.concat(obs_depth_list, dim="depth_sel").mean(dim="depth_sel", skipna=True)
    sim_avg = xr.concat(sim_depth_list, dim="depth_sel").mean(dim="depth_sel", skipna=True)

    # Monthly means, one value per year per month
    obs_monthly = obs_avg.resample(time="1M").mean(skipna=True)
    sim_monthly = sim_avg.resample(time="1M").mean(skipna=True)

    # Water-year order: Sep -> Aug
    month_order = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    month_labels = ["S", "O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A"]

    obs_mean, obs_min, obs_max = [], [], []
    sim_mean, sim_min, sim_max = [], [], []

    for m in month_order:
        ov = obs_monthly.sel(time=obs_monthly["time"].dt.month == m).values
        sv = sim_monthly.sel(time=sim_monthly["time"].dt.month == m).values
        ov = ov[~np.isnan(ov)]
        sv = sv[~np.isnan(sv)]

        obs_mean.append(np.mean(ov) if len(ov) else np.nan)
        obs_min.append(np.min(ov) if len(ov) else np.nan)
        obs_max.append(np.max(ov) if len(ov) else np.nan)

        sim_mean.append(np.mean(sv) if len(sv) else np.nan)
        sim_min.append(np.min(sv) if len(sv) else np.nan)
        sim_max.append(np.max(sv) if len(sv) else np.nan)

    x = np.arange(len(month_order))

    plt.figure(figsize=(8, 5))
    plt.plot(x, obs_mean, "-", color="gray", label="Observed", linewidth=2)
    plt.fill_between(x, obs_min, obs_max, color="gray", alpha=0.3)

    plt.plot(x, sim_mean, "-", color="crimson", label="Simulated", linewidth=2)
    plt.fill_between(x, sim_min, sim_max, color="crimson", alpha=0.3)

    plt.xticks(x, month_labels)
    plt.xlabel("Month")
    plt.ylabel(f"{vari}")
    plt.title(f"Seasonal Cycle of {vari} at {site} (depth-averaged)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"v1.0_final_images/{site}/{vari}/{site}_{vari}_seasonal.png")
    if not show:
        plt.close()