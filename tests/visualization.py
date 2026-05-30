import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

'''
def load_and_preprocess(file_path):
    """Loads sim_events.csv and performs basic preprocessing."""
    df = pd.read_csv(file_path)
    df['time_sec'] = df['timestamp'] / 1000.0
    return df
'''

def plot_throughput(df, window_size_sec=100):
    """Plots index construction throughput (points per second) over time."""
    finish_events = df[df['event_type'] == 'FINISH'].copy()
    finish_events = finish_events.sort_values('timestamp')
    
    max_time = int(finish_events['timestamp'].max())
    bins = range(0, max_time + window_size_sec, window_size_sec)
    finish_events['time_bin'] = pd.cut(finish_events['timestamp'], bins=bins, labels=bins[:-1])
    
    throughput = finish_events.groupby('time_bin', observed=True)['npts'].sum() / window_size_sec
    
    plt.figure()
    plt.plot(throughput.index.astype(float), throughput.values, marker='o', linestyle='-', markersize=4)
    plt.title('Index Construction Throughput Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Points / sec)')
    plt.savefig('tests/plot_throughput.png')
    plt.close()

'''
def plot_cumulative_progress(df):
    """Plots cumulative points completed over time (Burn-up chart)."""
    finish_events = df[df['event_type'] == 'FINISH'].copy()
    if finish_events.empty: return
    finish_events = finish_events.sort_values('time_sec')
    finish_events['cum_pts'] = finish_events['npts'].cumsum()
    
    plt.figure()
    plt.fill_between(finish_events['time_sec'], finish_events['cum_pts'], color="skyblue", alpha=0.4)
    plt.plot(finish_events['time_sec'], finish_events['cum_pts'], color="Slateblue", alpha=0.6)
    plt.title('Cumulative Work Progress (Points Completed)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Total Points Indexed')
    plt.savefig('tests/plot_cumulative_progress.png')
    plt.close()

def plot_resilience_overhead(df):
    """Plots task completion time vs number of interruptions."""
    task_stats = []
    for tid in df['task_id'].unique():
        tdf = df[df['task_id'] == tid]
        if 'FINISH' in tdf['event_type'].values:
            start_time = tdf['time_sec'].min()
            finish_time = tdf[tdf['event_type'] == 'FINISH']['time_sec'].max()
            interrupts = len(tdf[tdf['event_type'] == 'INTERRUPT'])
            task_stats.append({
                'task_id': tid,
                'duration': finish_time - start_time,
                'interruptions': interrupts
            })
    
    stats_df = pd.DataFrame(task_stats)
    if not stats_df.empty:
        plt.figure()
        sns.boxplot(x='interruptions', y='duration', data=stats_df)
        sns.stripplot(x='interruptions', y='duration', data=stats_df, color="orange", alpha=0.5)
        plt.title('Task Completion Duration vs. Number of Interruptions')
        plt.xlabel('Number of Interruptions')
        plt.ylabel('Wall-clock Time (seconds)')
        plt.savefig('tests/plot_resilience_overhead.png')
        plt.close()

def plot_concurrency(df):
    """Plots number of running tasks over time."""
    events = []
    for _, row in df.iterrows():
        if row['event_type'] == 'START':
            events.append((row['time_sec'], 1))
        elif row['event_type'] in ['FINISH', 'INTERRUPT']:
            events.append((row['time_sec'], -1))
    
    events.sort()
    times = [e[0] for e in events]
    counts = np.cumsum([e[1] for e in events])
    
    plt.figure()
    plt.step(times, counts, where='post')
    plt.title('System Concurrency (Active Tasks Over Time)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Concurrent Shard Builds')
    plt.savefig('tests/plot_concurrency.png')
    plt.close()

def plot_task_lifecycle(df, num_tasks=15):
    """Gantt chart showing task migrations across GPUs."""
    interrupt_counts = df[df['event_type'] == 'INTERRUPT'].groupby('task_id').size()
    interesting_tasks = interrupt_counts.sort_values(ascending=False).head(num_tasks).index
    
    if len(interesting_tasks) == 0:
        interesting_tasks = df['task_id'].unique()[:num_tasks]

    plt.figure(figsize=(12, 8))
    for i, tid in enumerate(interesting_tasks):
        tdf = df[df['task_id'] == tid].sort_values('time_sec')
        starts = tdf[tdf['event_type'] == 'START']
        
        for _, start_row in starts.iterrows():
            end_event = tdf[(tdf['time_sec'] > start_row['time_sec']) & 
                            (tdf['event_type'].isin(['FINISH', 'INTERRUPT']))].head(1)
            
            if not end_event.empty:
                duration = end_event.iloc[0]['time_sec'] - start_row['time_sec']
                plt.barh(i, duration, left=start_row['time_sec'], color=plt.cm.tab20(start_row['gpu_id'] % 20))
                plt.text(start_row['time_sec'] + duration/2, i, f"G{int(start_row['gpu_id'])}", 
                         ha='center', va='center', color='white', fontsize=8)

    plt.yticks(range(len(interesting_tasks)), interesting_tasks)
    plt.title(f'Task Migration Lifecycle (Top {num_tasks} Interrupted Tasks)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Task ID')
    plt.savefig('tests/plot_task_lifecycle.png')
    plt.close()

def plot_work_distribution(df):
    """Shows total points completed per GPU."""
    gpu_work = df[df['event_type'] == 'FINISH'].groupby('gpu_id')['npts'].sum().sort_values(ascending=False)
    
    if not gpu_work.empty:
        plt.figure()
        gpu_work.plot(kind='bar')
        plt.title('Work Distribution Across GPUs (Total Points Completed)')
        plt.xlabel('GPU ID')
        plt.ylabel('Total Points')
        plt.savefig('tests/plot_work_distribution.png')
        plt.close()

if __name__ == "__main__":
    csv_path = 'tests/sim_events.csv'
    if os.path.exists(csv_path):
        data = load_and_preprocess(csv_path)
        print("Generating plots...")
        plot_throughput(data)
        plot_cumulative_progress(data)
        plot_resilience_overhead(data)
        plot_concurrency(data)
        plot_task_lifecycle(data)
        plot_work_distribution(data)
        print("Done. Plots saved to tests/ directory.")
    else:
        print(f"Error: {csv_path} not found.")
'''
