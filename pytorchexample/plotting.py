import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TRIPLE_STYLE = dict(
    ber_ylim=(-0.05, 0.75),
    acc_ylim=(0, 100),
    ber_threshold=0.30,
    legend_fontsize=7,
    dpi=150,
    honest_color='C0',
    attacker_color='C3',
    def_color='C0',
    nodef_color='C1',
)

def triple_panel(run_nodef: str, run_def: str,
                 attack_name: str,
                 scale: float,
                 extra_title: str = '',
                 out_path: str = None,
                 has_asr: bool = False,
                 results_dir: str = 'results'):
    fig, (ax_ber_nodef, ax_ber_def, ax_acc) = plt.subplots(1, 3, figsize=(18, 5))
    s = TRIPLE_STYLE

    for mode, run_id, ax_ber in [("nodef", run_nodef, ax_ber_nodef),
                                  ("def", run_def, ax_ber_def)]:
        # Server metrics
        with open(f"{results_dir}/run_{run_id}_server.csv") as f:
            srows = list(csv.DictReader(f))
        rounds = [int(r['round']) for r in srows]
        accs = [float(r['server_acc']) * 100 for r in srows]

        ls = '-' if mode == 'def' else '--'
        color = s['def_color'] if mode == 'def' else s['nodef_color']
        ax_acc.plot(rounds, accs, ls, color=color, linewidth=1.5,
                    label=f'{"with def" if mode == "def" else "no def"}')

        if has_asr and 'server_asr' in srows[0]:
            asrs = [float(r['server_asr']) * 100 for r in srows]
            ax_acc.plot(rounds, asrs, ':', color=color, linewidth=1.0, alpha=0.6,
                        label=f'ASR {"w/ def" if mode == "def" else "no def"}')

        # Train metrics — separate honest / attacker BER
        with open(f"{results_dir}/run_{run_id}_train.csv") as f:
            trows = list(csv.DictReader(f))
        by_round = {}
        for r in trows:
            by_round.setdefault(int(r['round']), []).append(r)
        br = sorted(by_round)
        hm, hs, av = [], [], []
        for rnd in br:
            clients = by_round[rnd]
            hon = [float(c['watermark_ber']) for c in clients if not int(c['is_attacker'])]
            att = [float(c['watermark_ber']) for c in clients if int(c['is_attacker'])]
            hm.append(np.mean(hon) if hon else 0)
            hs.append(np.std(hon) if hon else 0)
            av.append(att[0] if att else np.nan)
        b = np.array(br)
        ax_ber.fill_between(b, np.array(hm) - np.array(hs),
                             np.array(hm) + np.array(hs), alpha=0.15, color=s['honest_color'])
        ax_ber.plot(b, np.array(hm), '-', color=s['honest_color'], linewidth=1.5, label='honest')
        att_valid = ~np.isnan(av)
        ax_ber.plot(b[att_valid], np.array(av)[att_valid], '--', color=s['attacker_color'],
                    linewidth=2, label='attacker', marker='s', markersize=3)
        ax_ber.axhline(y=s['ber_threshold'], color='gray', linestyle=':', alpha=0.5,
                       label=f'threshold={s["ber_threshold"]}')
        ax_ber.set_xlabel('Round')
        ax_ber.set_ylabel('Watermark BER')
        ax_ber.set_title(f'BER — {"No Defense" if mode == "nodef" else "With Defense"}')
        ax_ber.legend(fontsize=s['legend_fontsize'])
        ax_ber.grid(True, alpha=0.3)
        ax_ber.set_ylim(*s['ber_ylim'])

    ax_acc.set_xlabel('Round')
    ax_acc.set_ylabel('Accuracy (%)' + (' / ASR (%)' if has_asr else ''))
    ax_acc.set_title('Accuracy' + (' & ASR' if has_asr else '') + ' Comparison')
    ax_acc.legend(fontsize=s['legend_fontsize'])
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(*s['acc_ylim'])

    title = f'{attack_name} scale={scale} (50 rounds)'
    if extra_title:
        title += f' — {extra_title}'
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if out_path is None:
        out_path = f'{results_dir}/plots/{attack_name.lower().replace(" ", "_")}_sf{scale}_triple.png'
    fig.savefig(out_path, dpi=s['dpi'], bbox_inches='tight')
    plt.close(fig)
    return out_path


def regenerate_all(runs: dict, results_dir: str = 'results'):
    """Generate all triple plots from a runs dict.

    runs = {
        "noise": {
            "nodef": {0.1: "run_id", 0.25: "run_id", ...},
            "def":   {0.1: "run_id", 0.25: "run_id", ...},
        },
        "signflip": { ... },
        "labelflip": { ... },
    }
    """
    for attack_name, group in runs.items():
        for sc in sorted(group['def']):
            out = triple_panel(
                run_nodef=group['nodef'][sc],
                run_def=group['def'][sc],
                attack_name=attack_name.upper(),
                scale=sc,
                has_asr=(attack_name == 'labelflip'),
                results_dir=results_dir,
            )
            print(f"  {out}")
