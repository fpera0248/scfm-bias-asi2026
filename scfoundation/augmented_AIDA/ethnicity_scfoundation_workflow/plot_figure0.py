import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

OUT = pathlib.Path(
    "/oscar/home/fperalta/data/fperalta/scfoundation/augmented_AIDA/"
    "ethnicity_scfoundation_workflow/step9_visualizations_ethnicity/"
    "figure0_demographic_imbalance.png"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

ILD_SEX = {"Male": 287_679, "Female": 111_056}
ILD_AGE = {"10-19": 9_100, "20-29": 3_294, "30-39": 13_745, "40-49": 25_280,
           "50-59": 111_088, "60-69": 198_848, "70-79": 37_380}
ILD_ETH = {"European American": 333_602, "African American": 39_666,
           "Hispanic or Latin": 12_787, "Asian": 10_054, "Native American": 2_626}

CRC_SEX = {"Male": 17_737, "Female": 16_619}
CRC_AGE = {"30-39": 1_448, "40-49": 8_827, "50-59": 9_850,
           "60-69": 7_745, "70-79": 6_486}
CRC_ETH = {"European American": 6_456, "Hispanic or Latin": 1_111,
           "Asian": 619, "African American": 386}

SEX_COLORS = ["#4C72B0", "#C44E52"]
AGE_COLOR  = "#4C72B0"
ETH_COLORS = ["#FF8F00", "#7B1FA2", "#388E3C", "#1976D2", "#C44E52"]


def hbar(ax, data, title, colors=None, fontsize=7):
    labels = list(data.keys())
    values = list(data.values())
    if colors is None:
        colors = [AGE_COLOR] * len(labels)
    elif len(colors) < len(labels):
        colors = colors * len(labels)
    y = range(len(labels))
    ax.barh(y, values, color=colors[:len(labels)], edgecolor="white", linewidth=0.4)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=fontsize - 1, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold", pad=3)
    ax.set_xlabel("Cell count", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 1)
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, axes = plt.subplots(2, 3, figsize=(10, 4.5))

hbar(axes[0, 0], ILD_SEX, "ILD: Sex", SEX_COLORS)
hbar(axes[0, 1], ILD_AGE, "ILD: Age (decade bin)")
hbar(axes[0, 2], ILD_ETH, "ILD: Ethnicity", ETH_COLORS)
hbar(axes[1, 0], CRC_SEX, "CRC: Sex", SEX_COLORS)
hbar(axes[1, 1], CRC_AGE, "CRC: Age (decade bin)")
hbar(axes[1, 2], CRC_ETH, "CRC: Ethnicity", ETH_COLORS)

plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
fig.savefig(str(OUT), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
