#!/usr/bin/env python
"""Aggregate CTM analysis tables across multiple random-seed runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def aggregate_csv(paths, filename, group_cols, value_cols, out_path):
    frames=[]
    for p in paths:
        f=Path(p)/filename
        if f.exists():
            d=pd.read_csv(f)
            d['run']=Path(p).parent.name
            frames.append(d)
    if not frames:
        return None
    df=pd.concat(frames, ignore_index=True)
    vals=[c for c in value_cols if c in df.columns]
    agg=df.groupby(group_cols)[vals].agg(['mean','std']).reset_index()
    agg.columns=['_'.join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in agg.columns]
    agg.to_csv(out_path,index=False)
    return agg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis_dirs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    args=p.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    aggregate_csv(args.analysis_dirs,'per_tick_metrics.csv',['tick'],['accuracy','macro_f1','auc_ovr_macro','certainty','ece','brier'],out/'aggregate_per_tick.csv')
    aggregate_csv(args.analysis_dirs,'early_exit.csv',['threshold'],['accuracy','macro_f1','mean_ticks'],out/'aggregate_early_exit.csv')
    aggregate_csv(args.analysis_dirs,'difficulty_by_tick.csv',['difficulty','tick'],['accuracy','certainty'],out/'aggregate_difficulty.csv')
    aggregate_csv(args.analysis_dirs,'attention_event_bodies.csv',['class','ctm_tick'],['mean_attention_on_event_bodies','uniform_attention_baseline'],out/'aggregate_event_attention.csv')
    aggregate_csv(args.analysis_dirs,'representation_separation.csv',['tick'],['silhouette_by_outcome'],out/'aggregate_representation_separation.csv')
    aggregate_csv(args.analysis_dirs,'synchronization_by_tick.csv',['tick'],['mean_abs_output_sync'],out/'aggregate_synchronization.csv')
    print('Saved aggregate tables to',out)

if __name__=='__main__': main()
