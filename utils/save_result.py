import pandas as pd
import matplotlib.pyplot as plt


def print_results_table(results_by_version):
    """複数バージョンの結果をテーブルで表示"""
    df = pd.DataFrame(results_by_version).T
    df.index.name = 'Num Versions'

    print("\n=== 複数バージョンの結果比較 ===")
    print(df.to_string())
    print()


def plot_results(results_by_version):
    """複数バージョンの結果をグラフで表示"""
    versions = list(results_by_version.keys())

    # メトリクスごとにグラフを作成
    metrics = ['cov', 'cer', 'covod', 'cerod',
               'acc_affirmative', 'acc_majority', 'acc_unanimous']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results_by_version[v][metric] for v in versions]
        axes[idx].plot(versions, values, marker='o', linewidth=2, markersize=8)
        axes[idx].set_xlabel('Number of Versions')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} by Version')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(versions)

    # 最後のサブプロットは非表示
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()
