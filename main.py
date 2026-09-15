# ============================================================
# FULL PIPELINE — Data Preprocessing, EDA, Hybrid LightGBM-TabPFN
# Model, Flood-Risk Maps, AHP-GI Suitability, NSGA-II Optimization,
# and Evaluation Matrices (all ticks/labels/legends = FONT_SIZE 18, BOLD)
# ============================================================

import os,glob,re,rasterio,numpy as np,pandas as pd
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

BTH_ROOT=r"BTH data"
CLUD_ROOT=r"CLUD-Urban"
OUTPUT_CSV=r"BTH_CLUD_Integrated_Preprocessed.csv"
TARGET_COLUMN="FLOOD"
TRAIN_YEARS=[2000,2005]
SPATIAL_VARS=["DEM","HAND","PRE","WAT"]
SPATIAL_WINDOW=3
CHUNK_ROWS=128

input_matches=glob.glob(os.path.join(BTH_ROOT,"**","BTH_FloodRisk_Input_Data_2001_2025"),recursive=True)
if not input_matches: raise FileNotFoundError("BTH input folder not found")
BTH_INPUT=input_matches[0]

BTH_VARIABLES=["FLOOD","GDP","HAND","IMP","LAND","NDVI","PRE","SOIL","STD","WAT"]
CLUD_VARIABLES=["urban","ISA","UGS","WB"]

dem_files=[f for f in glob.glob(os.path.join(BTH_ROOT,"**","*.tif"),recursive=True) if os.path.basename(f).lower()=="dem.tif"]
if not dem_files: raise FileNotFoundError("DEM TIFF not found anywhere inside BTH dataset")
DEM_FILE=dem_files[0]

with rasterio.open(DEM_FILE) as src:
    REF_CRS=src.crs
    REF_TRANSFORM=src.transform
    REF_WIDTH=src.width
    REF_HEIGHT=src.height
    DEM_NODATA=src.nodata

print("BTH INPUT:",BTH_INPUT)
print("DEM:",DEM_FILE)
print("Reference grid:",REF_WIDTH,"x",REF_HEIGHT)
print("CRS:",REF_CRS)

def find_bth_file(variable,year):
    files=glob.glob(os.path.join(BTH_INPUT,"**","*.tif"),recursive=True)
    for f in files:
        name=os.path.basename(f).lower()
        if variable.lower() in name:
            m=re.search(r"(19|20)\d{2}",name)
            if m and int(m.group())==year:
                return f
    return None

def find_clud_file(variable,year):
    files=glob.glob(os.path.join(CLUD_ROOT,variable,"**","hdr.adf"),recursive=True)
    if not files:
        files=[f for f in glob.glob(os.path.join(CLUD_ROOT,"**","hdr.adf"),recursive=True) if variable.lower() in f.lower()]
    for f in files:
        folder=os.path.basename(os.path.dirname(f))
        if str(year) in folder:
            return f
    return None

bth_years=set()
for f in glob.glob(os.path.join(BTH_INPUT,"**","*.tif"),recursive=True):
    m=re.search(r"(19|20)\d{2}",os.path.basename(f))
    if m: bth_years.add(int(m.group()))

clud_years=set()
for variable in CLUD_VARIABLES:
    for f in glob.glob(os.path.join(CLUD_ROOT,variable,"**","hdr.adf"),recursive=True):
        m=re.search(r"(19|20)\d{2}",os.path.basename(os.path.dirname(f)))
        if m: clud_years.add(int(m.group()))

common_years=sorted(bth_years.intersection(clud_years))
if not common_years: raise ValueError("No common years found")
print("Common years:",common_years)
missing_train_years=[y for y in TRAIN_YEARS if y not in common_years]
if missing_train_years:
    raise ValueError(f"TRAIN_YEARS {missing_train_years} not found in common_years — fix TRAIN_YEARS")

year_arrays={}

for year in common_years:
    print("\nLoading year:",year)
    sources={}; clud_sources={}; vrts={}

    for variable in BTH_VARIABLES:
        f=find_bth_file(variable,year)
        if f: sources[variable]=rasterio.open(f)

    for variable in CLUD_VARIABLES:
        f=find_clud_file(variable,year)
        if f:
            src=rasterio.open(f)
            clud_sources[variable]=src
            vrts[variable]=WarpedVRT(src,crs=REF_CRS,transform=REF_TRANSFORM,width=REF_WIDTH,height=REF_HEIGHT,resampling=Resampling.nearest)

    dem_src=rasterio.open(DEM_FILE)
    dem=dem_src.read(1).astype(np.float32)
    if DEM_NODATA is not None: dem[dem==DEM_NODATA]=np.nan
    data={"DEM":dem}

    for variable,src in sources.items():
        arr=src.read(1).astype(np.float32)
        if src.nodata is not None: arr[arr==src.nodata]=np.nan
        data[variable]=arr

    for variable,vrt in vrts.items():
        arr=vrt.read(1).astype(np.float32)
        if vrt.nodata is not None: arr[arr==vrt.nodata]=np.nan
        data[variable]=arr

    dem_src.close()
    for src in sources.values(): src.close()
    for src in clud_sources.values(): src.close()
    for vrt in vrts.values(): vrt.close()

    year_arrays[year]=data
    print("Variables loaded:",list(data.keys()))

for year,data in year_arrays.items():
    for var in SPATIAL_VARS:
        if var in data:
            arr=data[var]
            valid=~np.isnan(arr)
            filled=np.where(valid,arr,0.0).astype(np.float32)
            local_sum=uniform_filter(filled,size=SPATIAL_WINDOW,mode="nearest")
            local_count=uniform_filter(valid.astype(np.float32),size=SPATIAL_WINDOW,mode="nearest")
            local_mean=np.divide(local_sum,local_count,out=np.full_like(local_sum,np.nan),where=local_count>0)
            data[f"{var}_NBR"]=local_mean.astype(np.float32)

feature_names=sorted(set(v for data in year_arrays.values() for v in data.keys()))
print("\nAll variables across years:",feature_names)

stats={}
for var in feature_names:
    vals=[year_arrays[y][var][~np.isnan(year_arrays[y][var])]
          for y in TRAIN_YEARS if var in year_arrays[y]]
    allvals=np.concatenate(vals) if vals else np.array([])
    if allvals.size>0:
        stats[var]={"mean":float(np.mean(allvals)),"min":float(np.min(allvals)),"max":float(np.max(allvals))}
    else:
        stats[var]={"mean":0.0,"min":0.0,"max":1.0}

if TARGET_COLUMN in feature_names:
    vals=[year_arrays[y][TARGET_COLUMN][~np.isnan(year_arrays[y][TARGET_COLUMN])]
          for y in TRAIN_YEARS if TARGET_COLUMN in year_arrays[y]]
    allvals=np.concatenate(vals) if vals else np.array([0])
    vals_int=np.round(allvals).astype(int) if allvals.size>0 else np.array([0])
    stats[TARGET_COLUMN]["mode"]=int(np.bincount(vals_int).argmax()) if vals_int.size>0 else 0

print("\nGlobal (train-year) stats computed for",len(stats),"variables")

if os.path.exists(OUTPUT_CSV): os.remove(OUTPUT_CSV)
header_written=False
total_rows=0
missing_before=0
missing_after=0

for year in common_years:
    data=year_arrays[year]
    feature_columns=list(data.keys())
    H,W=data["DEM"].shape

    missing_before+=sum(int(np.isnan(data[c]).sum()) for c in feature_columns)

    for c in feature_columns:
        nanmask=np.isnan(data[c])
        if nanmask.any():
            fill=stats[c].get("mode",stats[c]["mean"]) if c==TARGET_COLUMN else stats[c]["mean"]
            data[c]=np.where(nanmask,fill,data[c]).astype(np.float32)
        if c==TARGET_COLUMN:
            data[c]=np.round(data[c]).astype(np.float32)

    missing_after+=sum(int(np.isnan(data[c]).sum()) for c in feature_columns)

    for c in feature_columns:
        if c==TARGET_COLUMN: continue
        mn,mx=stats[c]["min"],stats[c]["max"]
        data[c]=(data[c]-mn)/(mx-mn) if mx>mn else np.zeros_like(data[c])

    data[TARGET_COLUMN]=(data[TARGET_COLUMN]>=0.5).astype(np.int8)

    for row_start in range(0,H,CHUNK_ROWS):
        row_end=min(row_start+CHUNK_ROWS,H)
        rows=row_end-row_start
        df_chunk=pd.DataFrame({
            "Year":np.full(rows*W,year,dtype=np.int16),
            "Row":np.repeat(np.arange(row_start,row_end),W),
            "Column":np.tile(np.arange(W),rows)
        })
        for c in feature_columns:
            df_chunk[c]=data[c][row_start:row_end,:].reshape(-1)
        df_chunk.to_csv(OUTPUT_CSV,mode="a",index=False,header=not header_written)
        header_written=True
        total_rows+=len(df_chunk)

    print(f"Year {year} written -> total rows: {total_rows:,}")
    del data

print("\n====================================================")
print("STEP 2 DATA PREPROCESSING COMPLETED")
print("====================================================")
print("Output file:",OUTPUT_CSV)
print("Common years:",common_years)
print("Total rows:",f"{total_rows:,}")
print("Missing values before imputation:",missing_before)
print("Missing values after imputation:",missing_after)
print("Normalization: GLOBAL min-max fit on TRAIN_YEARS",TRAIN_YEARS,"(no per-chunk drift, no leakage)")
print("Spatial features added:",[f"{v}_NBR" for v in SPATIAL_VARS])
print("====================================================")

SAMPLES_PER_YEAR=3000
SAMPLE_SEED=42
MAX_POSITIVE_FRACTION=0.5

df=pd.read_csv(OUTPUT_CSV)

def sample_per_year(frame,n_per_year,seed,max_positive_fraction=0.5):
    parts=[]
    for yr,group in frame.groupby("Year"):
        positives=group[group[TARGET_COLUMN]==1]
        negatives=group[group[TARGET_COLUMN]==0]
        max_positive_slots=max(int(n_per_year*max_positive_fraction),1)
        if len(positives)>max_positive_slots:
            kept_positives=positives.sample(n=max_positive_slots,random_state=seed)
        else:
            kept_positives=positives
        n_negatives_needed=n_per_year-len(kept_positives)
        n_negatives_take=min(n_negatives_needed,len(negatives))
        kept_negatives=negatives.sample(n=n_negatives_take,random_state=seed)
        year_sample=pd.concat([kept_positives,kept_negatives],ignore_index=True)
        parts.append(year_sample)
        print(f"Year {yr}: kept {len(kept_positives)} flood rows (of {len(positives)} available) "
              f"+ {len(kept_negatives)} no-flood rows -> {len(year_sample)} total")
    return pd.concat(parts,ignore_index=True)

print("\nRows before per-year sampling:",len(df))
df=sample_per_year(df,SAMPLES_PER_YEAR,SAMPLE_SEED,MAX_POSITIVE_FRACTION)
print("Rows after capping each year at",SAMPLES_PER_YEAR,"->",len(df))
print(df["Year"].value_counts().sort_index())
print("\nFlood-positive rate per year in sampled data:")
print(df.groupby("Year")[TARGET_COLUMN].mean())

SAMPLED_CSV="BTH_CLUD_Sampled_3000_Per_Year.csv"
df.to_csv(SAMPLED_CSV,index=False)
print("Sampled dataset saved to:",SAMPLED_CSV)

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("EDA_Results",exist_ok=True)

print("\nDataset Shape:",df.shape)
print("Years:",sorted(df["Year"].unique()))
print("Columns:",df.columns.tolist())
print("\nMissing Values:\n",df.isnull().sum())
print("\nDuplicate Rows:",df.duplicated().sum())
print("\nDescriptive Statistics:\n",df.describe())

features_all=["DEM","FLOOD","GDP","HAND","IMP","LAND","NDVI","PRE","SOIL","STD","WAT",
              "urban","ISA","UGS","WB","DEM_NBR","HAND_NBR","PRE_NBR","WAT_NBR"]
features_all=[c for c in features_all if c in df.columns]

stats_desc=df[features_all].describe().T
stats_desc["Range"]=stats_desc["max"]-stats_desc["min"]
stats_desc.to_csv("EDA_Results/Descriptive_Statistics.csv")

year_count=df["Year"].value_counts().sort_index()
print("\nSamples per Year:\n",year_count)
year_count.to_csv("EDA_Results/Yearwise_Sample_Count.csv")

flood_stats_year=df.groupby("Year")["FLOOD"].agg(["count","mean","std","min","max","median"])
print("\nYear-wise FLOOD Statistics:\n",flood_stats_year)
flood_stats_year.to_csv("EDA_Results/Yearwise_FLOOD_Statistics.csv")

corr=df[features_all].corr()
corr.to_csv("EDA_Results/Correlation_Matrix.csv")

flood_corr=corr["FLOOD"].drop("FLOOD").sort_values(key=abs,ascending=False)
print("\nFeature Correlation with FLOOD:\n",flood_corr)
flood_corr.to_csv("EDA_Results/FLOOD_Feature_Correlation.csv")

plt.figure(figsize=(8,5))
plt.bar(year_count.index.astype(str),year_count.values)
plt.xlabel("Year"); plt.ylabel("Number of Samples")
plt.title("Year-wise Sample Distribution")
plt.tight_layout(); plt.savefig("EDA_Results/Yearwise_Sample_Distribution.png",dpi=300); plt.close()

plt.figure(figsize=(8,5))
plt.hist(df["FLOOD"],bins=50)
plt.xlabel("FLOOD"); plt.ylabel("Frequency"); plt.title("FLOOD Distribution")
plt.tight_layout(); plt.savefig("EDA_Results/FLOOD_Distribution.png",dpi=300); plt.close()

plt.figure(figsize=(8,5))
plt.plot(flood_stats_year.index,flood_stats_year["mean"],marker="o")
plt.xlabel("Year"); plt.ylabel("Mean FLOOD"); plt.title("Year-wise FLOOD Trend")
plt.xticks(flood_stats_year.index)
plt.tight_layout(); plt.savefig("EDA_Results/FLOOD_Yearwise_Trend.png",dpi=300); plt.close()

plt.figure(figsize=(12,9))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout(); plt.savefig("EDA_Results/Correlation_Matrix.png",dpi=300); plt.close()

for col in features_all:
    if col!="FLOOD":
        plt.figure(figsize=(6,4))
        plt.hist(df[col],bins=40)
        plt.xlabel(col); plt.ylabel("Frequency"); plt.title(f"{col} Distribution")
        plt.tight_layout(); plt.savefig(f"EDA_Results/Distribution_{col}.png",dpi=300); plt.close()

year_mean=df.groupby("Year")[features_all].mean()
year_mean.to_csv("EDA_Results/Yearwise_Feature_Mean.csv")

print("\n================ EDA COMPLETED ================")
print("Dataset:",df.shape)
print("Years:",sorted(df["Year"].unique()))
print("Missing values:",df.isnull().sum().sum())
print("Duplicate rows:",df.duplicated().sum())
print("EDA results saved in: EDA_Results")
print("================================================")

target="FLOOD"

features=[
    "DEM","GDP","HAND","IMP","LAND","NDVI","PRE","SOIL","STD","WAT",
    "urban","ISA","UGS","WB",
    "DEM_NBR","HAND_NBR","PRE_NBR","WAT_NBR"
]

missing_features=[c for c in features if c not in df.columns]
if missing_features:
    print("\nWARNING: these requested features were not found in the data and will be dropped:",missing_features)
features=[c for c in features if c in df.columns]

def _safe_div(a,b,eps=1e-6):
    return a/(b+eps)

if "HAND" in df.columns and "DEM" in df.columns:
    df["HAND_DEM"]=df["HAND"]*df["DEM"]
    df["HAND_DEM_ratio"]=_safe_div(df["HAND"],df["DEM"])
    features+=["HAND_DEM","HAND_DEM_ratio"]

if "PRE" in df.columns and "WAT" in df.columns:
    df["PRE_WAT"]=df["PRE"]*df["WAT"]
    features.append("PRE_WAT")

if "IMP" in df.columns and "PRE" in df.columns:
    df["IMP_PRE"]=df["IMP"]*df["PRE"]
    features.append("IMP_PRE")

if "NDVI" in df.columns:
    df["NDVI_neg"]=1.0-df["NDVI"]
    features.append("NDVI_neg")

if "PRE" in df.columns and "HAND" in df.columns:
    df["PRE_HAND"]=df["PRE"]*(1.0-df["HAND"])
    features.append("PRE_HAND")

if "PRE_NBR" in df.columns and "HAND_NBR" in df.columns:
    df["PRE_HAND_NBR"]=df["PRE_NBR"]*(1.0-df["HAND_NBR"])
    features.append("PRE_HAND_NBR")

_base=["DEM","GDP","HAND","IMP","LAND","NDVI","PRE","SOIL","STD","WAT","urban","ISA","UGS","WB",
       "DEM_NBR","HAND_NBR","PRE_NBR","WAT_NBR"]
_eng=[f for f in features if f not in _base]
print("Engineered features added:",_eng)
print("Total features after engineering:",len(features))

X=df[features].copy()
y=df[target].copy()

df["FLOOD_PREDICTED"]=np.nan
df["FLOOD_PROBABILITY"]=np.nan

print("Target variable:",target)
print("Features:",features)
print("X shape:",X.shape)
print("y shape:",y.shape)
print("Actual FLOOD unique values:",sorted(df["FLOOD"].dropna().unique())[:20])

available_years=sorted(df["Year"].dropna().unique())
print("\nAvailable years:",available_years)

TRAIN_YEARS_MODEL=[2000,2005]
TEST_YEAR=2010
VAL_YEAR=2015

train_df=df[df["Year"].isin(TRAIN_YEARS_MODEL)].copy()
test_df=df[df["Year"]==TEST_YEAR].copy()
val_df=df[df["Year"]==VAL_YEAR].copy()

X_train=train_df[features].copy(); y_train=train_df[target].copy()
X_test=test_df[features].copy(); y_test=test_df[target].copy()
X_val=val_df[features].copy(); y_val=val_df[target].copy()

train_indices=train_df.index; test_indices=test_df.index; val_indices=val_df.index

print("\nSPLIT SUMMARY")
print("Training years:",TRAIN_YEARS_MODEL,"-> X_train:",X_train.shape)
print("Testing year:",TEST_YEAR,"-> X_test:",X_test.shape)
print("Validation year:",VAL_YEAR,"-> X_val:",X_val.shape)

import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    roc_auc_score,average_precision_score,confusion_matrix,matthews_corrcoef,classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

print("\n============================================================")
print("STEP 7 — HYBRID LIGHTGBM–TabPFN")
print("============================================================")
print("Target:",target); print("Features:",features)

def to_binary(series):
    return (series.astype(float)>=0.5).astype(int)

y_train=to_binary(y_train); y_test=to_binary(y_test); y_val=to_binary(y_val)

unique_target=sorted(pd.Series(y_train).dropna().unique())
print("Training target values:",unique_target)
if len(unique_target)!=2:
    raise ValueError("FLOOD is not binary after thresholding. Found: "+str(unique_target))

print("\n============================================================")
print("LightGBM training (per-round logging)")
print("Train: 2000 + 2005 | Early-stop monitor: 2015 (validation)")
print("============================================================")

train_class_counts=y_train.value_counts()
neg_count=int(train_class_counts.get(0,0))
pos_count=int(train_class_counts.get(1,1))
scale_pos_weight=neg_count/pos_count if pos_count>0 else 1.0
print("Train class counts -> No Flood:",neg_count,"| Flood:",pos_count)
print("scale_pos_weight used:",round(scale_pos_weight,3))

lgbm_model=LGBMClassifier(
    objective="binary",n_estimators=5000,learning_rate=0.01,num_leaves=63,
    max_depth=-1,subsample=0.85,subsample_freq=5,colsample_bytree=0.8,
    reg_alpha=0.2,reg_lambda=1.0,min_child_samples=15,min_split_gain=0.0,
    scale_pos_weight=scale_pos_weight,random_state=42,n_jobs=-1,verbosity=-1
)

lgbm_evals_result={}
lgbm_model.fit(
    X_train,y_train,
    eval_set=[(X_train,y_train),(X_val,y_val)],
    eval_names=["train","validation"],
    eval_metric=["binary_logloss","auc"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150,verbose=True),
        lgb.record_evaluation(lgbm_evals_result),
        lgb.log_evaluation(period=100)
    ]
)
print("\nBest boosting round (early-stopped):",lgbm_model.best_iteration_)

lgbm_train_prob=lgbm_model.predict_proba(X_train)[:,1]
lgbm_val_prob=lgbm_model.predict_proba(X_val)[:,1]
lgbm_test_prob=lgbm_model.predict_proba(X_test)[:,1]

importance_df=pd.DataFrame({"Feature":features,"Importance":lgbm_model.feature_importances_}).sort_values("Importance",ascending=False)
print("\nLightGBM Feature Importance:\n",importance_df)
importance_df.to_csv("LightGBM_Feature_Importance.csv",index=False)

print("\n============================================================")
print("HYBRID META-FEATURE CONSTRUCTION")
print("============================================================")

X_train_hybrid=X_train.copy(); X_val_hybrid=X_val.copy(); X_test_hybrid=X_test.copy()
X_train_hybrid["LightGBM_Probability"]=lgbm_train_prob
X_val_hybrid["LightGBM_Probability"]=lgbm_val_prob
X_test_hybrid["LightGBM_Probability"]=lgbm_test_prob

print("\n============================================================")
print("TabPFN TRAINING")
print("============================================================")

def balance_classes(X_frame,y_series,seed=42):
    combined=X_frame.copy()
    combined["_target_"]=y_series.values
    counts=combined["_target_"].value_counts()
    max_count=counts.max()
    parts=[]
    for cls,count in counts.items():
        part=combined[combined["_target_"]==cls]
        if count<max_count:
            part=part.sample(n=max_count,replace=True,random_state=seed)
        parts.append(part)
    balanced=pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)
    y_balanced=balanced["_target_"]
    X_balanced=balanced.drop(columns=["_target_"])
    return X_balanced,y_balanced

X_train_balanced,y_train_balanced=balance_classes(X_train_hybrid,y_train,seed=42)
print("Before balancing:",y_train.value_counts().to_dict())
print("After balancing:",y_train_balanced.value_counts().to_dict())

MAX_TABPFN_TRAIN=4000
if len(X_train_balanced)>MAX_TABPFN_TRAIN:
    rng=np.random.RandomState(42)
    sample_indices=rng.choice(len(X_train_balanced),size=MAX_TABPFN_TRAIN,replace=False)
    X_tabpfn_train=X_train_balanced.iloc[sample_indices].copy()
    y_tabpfn_train=y_train_balanced.iloc[sample_indices].copy()
else:
    X_tabpfn_train=X_train_balanced.copy()
    y_tabpfn_train=y_train_balanced.copy()
print("TabPFN training samples:",len(X_tabpfn_train))

tabpfn_model=TabPFNClassifier(random_state=42,ignore_pretraining_limits=True)
tabpfn_model.fit(X_tabpfn_train,y_tabpfn_train)

tabpfn_val_prob=tabpfn_model.predict_proba(X_val_hybrid)[:,1]
tabpfn_test_prob=tabpfn_model.predict_proba(X_test_hybrid)[:,1]

print("\n============================================================")
print("5-FOLD OOF STACKING META-LEARNER")
print("============================================================")

N_FOLDS=5
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=42)
X_train_r=X_train.reset_index(drop=True)
y_train_r=y_train.reset_index(drop=True)

lgbm_oof=np.zeros(len(X_train_r))
tabpfn_oof=np.zeros(len(X_train_r))

for fold,(idx_tr,idx_vf) in enumerate(skf.split(X_train_r,y_train_r)):
    print(f"  Fold {fold+1}/{N_FOLDS}")
    Xf_tr=X_train_r.iloc[idx_tr].copy(); Xf_vf=X_train_r.iloc[idx_vf].copy()
    yf_tr=y_train_r.iloc[idx_tr]; yf_vf=y_train_r.iloc[idx_vf]

    cnt_f=yf_tr.value_counts()
    spw_f=int(cnt_f.get(0,1))/max(int(cnt_f.get(1,1)),1)

    lgbm_f=LGBMClassifier(
        objective="binary",n_estimators=5000,learning_rate=0.01,num_leaves=63,
        max_depth=-1,subsample=0.85,subsample_freq=5,colsample_bytree=0.8,
        reg_alpha=0.2,reg_lambda=1.0,min_child_samples=15,min_split_gain=0.0,
        scale_pos_weight=spw_f,random_state=42,n_jobs=-1,verbosity=-1
    )
    lgbm_f.fit(Xf_tr,yf_tr,eval_set=[(Xf_vf,yf_vf)],
               callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(0)])
    lgbm_oof[idx_vf]=lgbm_f.predict_proba(Xf_vf)[:,1]

    lgbm_f_tr_p=lgbm_f.predict_proba(Xf_tr)[:,1]
    lgbm_f_vf_p=lgbm_f.predict_proba(Xf_vf)[:,1]
    Xf_tr_h=Xf_tr.copy(); Xf_tr_h["LightGBM_Probability"]=lgbm_f_tr_p
    Xf_vf_h=Xf_vf.copy(); Xf_vf_h["LightGBM_Probability"]=lgbm_f_vf_p

    Xf_bal,yf_bal=balance_classes(Xf_tr_h,yf_tr,seed=42)
    if len(Xf_bal)>MAX_TABPFN_TRAIN:
        idx_s=np.random.RandomState(42).choice(len(Xf_bal),MAX_TABPFN_TRAIN,replace=False)
        Xf_bal=Xf_bal.iloc[idx_s].copy(); yf_bal=yf_bal.iloc[idx_s]

    tab_f=TabPFNClassifier(random_state=42,ignore_pretraining_limits=True)
    tab_f.fit(Xf_bal,yf_bal)
    tabpfn_oof[idx_vf]=tab_f.predict_proba(Xf_vf_h)[:,1]

print("OOF LightGBM AUC:",round(roc_auc_score(y_train_r,lgbm_oof),4))
print("OOF TabPFN   AUC:",round(roc_auc_score(y_train_r,tabpfn_oof),4))

meta_X_tr=np.column_stack([lgbm_oof,tabpfn_oof])
meta_scaler=StandardScaler()
meta_X_tr_sc=meta_scaler.fit_transform(meta_X_tr)
meta_learner=LogisticRegression(C=1.0,random_state=42,max_iter=1000)
meta_learner.fit(meta_X_tr_sc,y_train_r)
print("Meta-learner coefficients [LGBM, TabPFN]:",meta_learner.coef_.round(4))

meta_X_val=meta_scaler.transform(np.column_stack([lgbm_val_prob,tabpfn_val_prob]))
meta_X_test=meta_scaler.transform(np.column_stack([lgbm_test_prob,tabpfn_test_prob]))

hybrid_val_probability=meta_learner.predict_proba(meta_X_val)[:,1]
hybrid_test_probability=meta_learner.predict_proba(meta_X_test)[:,1]

print("Hybrid (meta) val  AUC:",round(roc_auc_score(y_val,hybrid_val_probability),4))
print("Hybrid (meta) test AUC:",round(roc_auc_score(y_test,hybrid_test_probability),4))

def find_best_threshold_gmean(y_true,probability):
    best_threshold=0.5; best_score=-1.0
    for t in np.arange(0.05,0.96,0.005):
        pred=(probability>=t).astype(int)
        tn_t,fp_t,fn_t,tp_t=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
        sens=tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0.0
        spec=tn_t/(tn_t+fp_t) if (tn_t+fp_t)>0 else 0.0
        gmean=np.sqrt(sens*spec)
        if gmean>best_score:
            best_score=gmean; best_threshold=round(float(t),3)
    return best_threshold,best_score

lgbm_threshold,lgbm_val_gmean=find_best_threshold_gmean(y_val,lgbm_val_prob)
tabpfn_threshold,tabpfn_val_gmean=find_best_threshold_gmean(y_val,tabpfn_val_prob)
hybrid_threshold,hybrid_val_gmean=find_best_threshold_gmean(y_val,hybrid_val_probability)

print("\n============================================================")
print("THRESHOLD SELECTION — G-mean (Validation 2015, applied to Test 2010)")
print("============================================================")
print("LightGBM threshold:",lgbm_threshold,"| Val G-mean:",round(lgbm_val_gmean,4))
print("TabPFN   threshold:",tabpfn_threshold,"| Val G-mean:",round(tabpfn_val_gmean,4))
print("Hybrid   threshold:",hybrid_threshold,"| Val G-mean:",round(hybrid_val_gmean,4))

lgbm_val_pred=(lgbm_val_prob>=lgbm_threshold).astype(int)
lgbm_test_pred=(lgbm_test_prob>=lgbm_threshold).astype(int)
tabpfn_val_pred=(tabpfn_val_prob>=tabpfn_threshold).astype(int)
tabpfn_test_pred=(tabpfn_test_prob>=tabpfn_threshold).astype(int)
hybrid_val_prediction=(hybrid_val_probability>=hybrid_threshold).astype(int)
hybrid_test_prediction=(hybrid_test_probability>=hybrid_threshold).astype(int)

def evaluate_model(name,y_true,probability,prediction):
    tn,fp,fn,tp=confusion_matrix(y_true,prediction,labels=[0,1]).ravel()
    accuracy=accuracy_score(y_true,prediction)
    precision=precision_score(y_true,prediction,zero_division=0)
    recall=recall_score(y_true,prediction,zero_division=0)
    f1=f1_score(y_true,prediction,zero_division=0)
    roc_auc=roc_auc_score(y_true,probability)
    pr_auc=average_precision_score(y_true,probability)
    specificity=tn/(tn+fp) if (tn+fp)>0 else 0
    mcc=matthews_corrcoef(y_true,prediction)
    return {"Model":name,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-Score":f1,
            "ROC-AUC":roc_auc,"PR-AUC":pr_auc,"Specificity":specificity,"MCC":mcc,
            "TN":tn,"FP":fp,"FN":fn,"TP":tp}

lgbm_val_results=evaluate_model("LightGBM (Validation 2015)",y_val,lgbm_val_prob,lgbm_val_pred)
tabpfn_val_results=evaluate_model("TabPFN (Validation 2015)",y_val,tabpfn_val_prob,tabpfn_val_pred)
hybrid_val_results=evaluate_model("Hybrid LightGBM-TabPFN (Validation 2015)",y_val,hybrid_val_probability,hybrid_val_prediction)
lgbm_test_results=evaluate_model("LightGBM (Test 2010)",y_test,lgbm_test_prob,lgbm_test_pred)
tabpfn_test_results=evaluate_model("TabPFN (Test 2010)",y_test,tabpfn_test_prob,tabpfn_test_pred)
hybrid_test_results=evaluate_model("Hybrid LightGBM-TabPFN (Test 2010)",y_test,hybrid_test_probability,hybrid_test_prediction)

results_df=pd.DataFrame([lgbm_val_results,tabpfn_val_results,hybrid_val_results,
                          lgbm_test_results,tabpfn_test_results,hybrid_test_results])

print("\n============================================================")
print("FINAL MODEL PERFORMANCE — VALIDATION (2015) & TEST (2010)")
print("============================================================")
print(results_df[["Model","Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC","Specificity","MCC"]].round(4).to_string(index=False))

print("\nHYBRID CLASSIFICATION REPORT — TEST (2010)")
print(classification_report(y_test,hybrid_test_prediction,target_names=["No Flood","Flood"],digits=4,zero_division=0))
print("\nHYBRID CLASSIFICATION REPORT — VALIDATION (2015)")
print(classification_report(y_val,hybrid_val_prediction,target_names=["No Flood","Flood"],digits=4,zero_division=0))

print("\nConfusion Matrix — Test (2010):\n",confusion_matrix(y_test,hybrid_test_prediction))
print("\nConfusion Matrix — Validation (2015):\n",confusion_matrix(y_val,hybrid_val_prediction))

results_df.to_csv("Hybrid_LightGBM_TabPFN_Performance.csv",index=False)

df.loc[test_indices,"FLOOD_PREDICTED"]=hybrid_test_prediction
df.loc[test_indices,"FLOOD_PROBABILITY"]=hybrid_test_probability
df.loc[test_indices,"LIGHTGBM_PROBABILITY"]=lgbm_test_prob
df.loc[test_indices,"TABPFN_PROBABILITY"]=tabpfn_test_prob

df.loc[val_indices,"FLOOD_PREDICTED"]=hybrid_val_prediction
df.loc[val_indices,"FLOOD_PROBABILITY"]=hybrid_val_probability
df.loc[val_indices,"LIGHTGBM_PROBABILITY"]=lgbm_val_prob
df.loc[val_indices,"TABPFN_PROBABILITY"]=tabpfn_val_prob

test_output=test_df[["Year","Row","Column"]].copy()
test_output["FLOOD_ACTUAL"]=y_test.values
test_output["LIGHTGBM_PROBABILITY"]=lgbm_test_prob
test_output["TABPFN_PROBABILITY"]=tabpfn_test_prob
test_output["FLOOD_PROBABILITY"]=hybrid_test_probability
test_output["FLOOD_PREDICTED"]=hybrid_test_prediction
test_output.to_csv("2010_Hybrid_Flood_Test_Predictions.csv",index=False)

val_output=val_df[["Year","Row","Column"]].copy()
val_output["FLOOD_ACTUAL"]=y_val.values
val_output["LIGHTGBM_PROBABILITY"]=lgbm_val_prob
val_output["TABPFN_PROBABILITY"]=tabpfn_val_prob
val_output["FLOOD_PROBABILITY"]=hybrid_val_probability
val_output["FLOOD_PREDICTED"]=hybrid_val_prediction
val_output.to_csv("2015_Hybrid_Flood_Validation_Predictions.csv",index=False)

print("\n============================================================")
print("STEP 7 + STEP 8 COMPLETED")
print("============================================================")
print("Train years:",TRAIN_YEARS_MODEL,"| Test year:",TEST_YEAR,"| Validation year:",VAL_YEAR)
print("Samples per year (capped):",SAMPLES_PER_YEAR)
print("Decision thresholds: LightGBM->",lgbm_threshold,"| TabPFN->",tabpfn_threshold,"| Hybrid->",hybrid_threshold)


# ============================================================
# STEP 9 (REVISED) — 1500-point sample for 2015, maps show the
# actual BTH study-area SHAPE as background, continuous COLORMAP
# raster (interpolated, not scatter). All ticks/labels = FONT_SIZE 18, BOLD.
# ============================================================

import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

OUTPUT_FOLDER="Final_Maps"
os.makedirs(OUTPUT_FOLDER,exist_ok=True)

MAP_YEAR=2015
SAMPLE_SIZE=1500
LOW_THRESH=0.33
HIGH_THRESH=0.66

FIGSIZE=(10,8)
FONT_FAMILY="Times New Roman"
FONT_SIZE=18
FONT_WEIGHT="bold"
DPI=800
GRID_DOWNSAMPLE=2

plt.rcParams["font.family"]=FONT_FAMILY
plt.rcParams["font.size"]=FONT_SIZE

def style_axes(ax,title,xlabel="Column",ylabel="Row"):
    ax.set_title(title,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.set_xlabel(xlabel,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.grid(False)
    ax.tick_params(labelsize=FONT_SIZE)
    # BOLD tick numbers (0, 100, 200...) in addition to bold title/axis labels
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontfamily(FONT_FAMILY); lbl.set_fontsize(FONT_SIZE); lbl.set_fontweight(FONT_WEIGHT)

def save_map(fig,filename):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER,filename),dpi=DPI)
    plt.show()
    plt.close(fig)

print("\n============================================================")
print(f"STEP 9 — Flood-risk map generation ({SAMPLE_SIZE}-point sample, year {MAP_YEAR})")
print("============================================================")

with rasterio.open(DEM_FILE) as ref_src:
    REF_CRS=ref_src.crs
    REF_TRANSFORM=ref_src.transform
    REF_WIDTH=ref_src.width
    REF_HEIGHT=ref_src.height
    dem_boundary=ref_src.read(1).astype(np.float32)
    boundary_nodata=ref_src.nodata

if boundary_nodata is not None:
    BOUNDARY_MASK=(dem_boundary!=boundary_nodata) & ~np.isnan(dem_boundary)
else:
    BOUNDARY_MASK=~np.isnan(dem_boundary)
print("Study-area valid pixels:",int(BOUNDARY_MASK.sum()),"of",REF_HEIGHT*REF_WIDTH)

GRID_H=REF_HEIGHT//GRID_DOWNSAMPLE
GRID_W=REF_WIDTH//GRID_DOWNSAMPLE
grid_rows,grid_cols=np.mgrid[0:REF_HEIGHT:complex(GRID_H),0:REF_WIDTH:complex(GRID_W)]
GRID_MASK=BOUNDARY_MASK[grid_rows.astype(int).clip(0,REF_HEIGHT-1),
                         grid_cols.astype(int).clip(0,REF_WIDTH-1)]

def _interp_to_grid(row_idx,col_idx,values,method):
    pts=np.column_stack([np.asarray(row_idx,dtype=float),np.asarray(col_idx,dtype=float)])
    grid_vals=griddata(pts,np.asarray(values,dtype=float),(grid_rows,grid_cols),method=method)
    if method!="nearest":
        nan_holes=np.isnan(grid_vals)
        if nan_holes.any():
            fallback=griddata(pts,np.asarray(values,dtype=float),(grid_rows,grid_cols),method="nearest")
            grid_vals[nan_holes]=fallback[nan_holes]
    grid_vals=np.where(GRID_MASK,grid_vals,np.nan)
    return grid_vals

def plot_raster_map(row_idx,col_idx,values,title,filename,cmap="viridis",categorical=False,cat_labels=None,vmin=None,vmax=None,point_size=55):
    fig,ax=plt.subplots(figsize=FIGSIZE)

    background=np.where(BOUNDARY_MASK,np.nan,1.0)
    ax.imshow(background,cmap=mcolors.ListedColormap(["#ffffff"]),
              extent=[0,REF_WIDTH,REF_HEIGHT,0],aspect="equal",zorder=0)

    if categorical:
        unique_vals=sorted(pd.unique(values))
        cmap_obj=plt.get_cmap(cmap,len(unique_vals))
        val_to_idx={v:i for i,v in enumerate(unique_vals)}
        color_idx_vals=np.array([val_to_idx[v] for v in values],dtype=float)
        grid_vals=_interp_to_grid(row_idx,col_idx,color_idx_vals,method="nearest")
        im=ax.imshow(grid_vals,cmap=cmap_obj,vmin=-0.5,vmax=len(unique_vals)-0.5,
                      extent=[0,REF_WIDTH,REF_HEIGHT,0],aspect="equal",zorder=1,
                      interpolation="nearest")
        cbar=fig.colorbar(im,ax=ax,ticks=range(len(unique_vals)))
        if cat_labels is not None:
            try:
                tick_labels=[cat_labels[int(v)] for v in unique_vals]
            except (IndexError,TypeError,ValueError):
                tick_labels=[str(v) for v in unique_vals]
        else:
            tick_labels=[str(v) for v in unique_vals]
        cbar.ax.set_yticklabels(tick_labels,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
        cbar.ax.tick_params(labelsize=FONT_SIZE)
    else:
        grid_vals=_interp_to_grid(row_idx,col_idx,values,method="cubic")
        im=ax.imshow(grid_vals,cmap=cmap,vmin=vmin,vmax=vmax,
                      extent=[0,REF_WIDTH,REF_HEIGHT,0],aspect="equal",zorder=1,
                      interpolation="bilinear")
        cbar=fig.colorbar(im,ax=ax)
        cbar.ax.tick_params(labelsize=FONT_SIZE)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontfamily(FONT_FAMILY); lbl.set_fontsize(FONT_SIZE); lbl.set_fontweight(FONT_WEIGHT)

    ax.set_xlim(0,REF_WIDTH)
    ax.set_ylim(REF_HEIGHT,0)
    ax.set_facecolor("white")
    style_axes(ax,title)
    save_map(fig,filename)

full_df=pd.read_csv(OUTPUT_CSV)
full_year_df=full_df[full_df["Year"]==MAP_YEAR].copy()
if full_year_df.empty:
    raise ValueError(f"No rows found for year {MAP_YEAR} in {OUTPUT_CSV}")
print("Full raster rows available for",MAP_YEAR,":",len(full_year_df))

df_2015=full_year_df.sample(n=min(SAMPLE_SIZE,len(full_year_df)),random_state=42).reset_index(drop=True)
print("Rows sampled for analysis:",len(df_2015))

if "HAND" in df_2015.columns and "DEM" in df_2015.columns:
    df_2015["HAND_DEM"]=df_2015["HAND"]*df_2015["DEM"]
    df_2015["HAND_DEM_ratio"]=_safe_div(df_2015["HAND"],df_2015["DEM"])
if "PRE" in df_2015.columns and "WAT" in df_2015.columns:
    df_2015["PRE_WAT"]=df_2015["PRE"]*df_2015["WAT"]
if "IMP" in df_2015.columns and "PRE" in df_2015.columns:
    df_2015["IMP_PRE"]=df_2015["IMP"]*df_2015["PRE"]
if "NDVI" in df_2015.columns:
    df_2015["NDVI_neg"]=1.0-df_2015["NDVI"]
if "PRE" in df_2015.columns and "HAND" in df_2015.columns:
    df_2015["PRE_HAND"]=df_2015["PRE"]*(1.0-df_2015["HAND"])
if "PRE_NBR" in df_2015.columns and "HAND_NBR" in df_2015.columns:
    df_2015["PRE_HAND_NBR"]=df_2015["PRE_NBR"]*(1.0-df_2015["HAND_NBR"])

missing_in_sample=[c for c in features if c not in df_2015.columns]
if missing_in_sample:
    raise ValueError("Sample frame is missing required features: "+str(missing_in_sample))

with rasterio.open(DEM_FILE) as dem_src2:
    dem_full=dem_src2.read(1).astype(np.float32)
    res_x=dem_src2.transform[0]
    res_y=-dem_src2.transform[4]
if DEM_NODATA is not None:
    dem_full[dem_full==DEM_NODATA]=np.nan
dzdy,dzdx=np.gradient(dem_full,res_y,res_x)
slope_full=np.degrees(np.arctan(np.sqrt(dzdx**2+dzdy**2)))
df_2015["Slope"]=slope_full[df_2015["Row"].values.astype(int),df_2015["Column"].values.astype(int)]
df_2015["Slope"]=df_2015["Slope"].fillna(df_2015["Slope"].mean())

X_sample=df_2015[features].copy()
lgbm_sample_prob=lgbm_model.predict_proba(X_sample)[:,1]
X_sample_hybrid=X_sample.copy()
X_sample_hybrid["LightGBM_Probability"]=lgbm_sample_prob
tabpfn_sample_prob=tabpfn_model.predict_proba(X_sample_hybrid)[:,1]
meta_X_sample=meta_scaler.transform(np.column_stack([lgbm_sample_prob,tabpfn_sample_prob]))
hybrid_sample_probability=meta_learner.predict_proba(meta_X_sample)[:,1]
hybrid_sample_prediction=(hybrid_sample_probability>=hybrid_threshold).astype(np.int8)

df_2015["FLOOD_PROBABILITY"]=hybrid_sample_probability
df_2015["FLOOD_PREDICTED"]=hybrid_sample_prediction

def classify_risk(p):
    if p<LOW_THRESH: return 0
    elif p<HIGH_THRESH: return 1
    else: return 2

df_2015["RISK_ZONE"]=df_2015["FLOOD_PROBABILITY"].apply(classify_risk).astype(np.int8)

df_2015.to_csv(os.path.join(OUTPUT_FOLDER,f"{MAP_YEAR}_Sampled_Flood_Predictions.csv"),index=False)

plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["FLOOD"],
    f"Observed Flood Map — {MAP_YEAR}","01_Observed_Flood_Map.png",
    cmap=mcolors.ListedColormap(["#2ca25f","#d73027"]),categorical=True,cat_labels=["No Flood","Flood"])

plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["FLOOD_PROBABILITY"],
    f"Predicted Flood Probability — {MAP_YEAR}","02_Predicted_Flood_Probability_Map.png",
    cmap="RdYlGn_r",vmin=0,vmax=1)

risk_cmap=mcolors.ListedColormap(["#2ca25f","#fee08b","#d73027"])
plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["RISK_ZONE"],
    f"Flood-Risk Zones — {MAP_YEAR}","03_Flood_Risk_Zone_Map.png",
    cmap=risk_cmap,categorical=True,cat_labels=["Low","Medium","High"])

print("\n============================================================")
print("STEP 10 — AHP GI SUITABILITY ANALYSIS")
print("============================================================")

AHP_CRITERIA=["Flood_Probability","UGS_Need","Impervious_Surface","Water_Context",
              "Elevation_Suitability","Slope_Suitability","Urban_Presence"]

AHP_MATRIX=np.array([
    [1,   5,   2,   5,   3,   7,   3  ],
    [1/5, 1,   1/3, 1,   1/3, 2,   1/2],
    [1/2, 3,   1,   3,   1,   4,   2  ],
    [1/5, 1,   1/3, 1,   1/2, 2,   1  ],
    [1/3, 3,   1,   2,   1,   3,   1  ],
    [1/7, 1/2, 1/4, 1/2, 1/3, 1,   1/2],
    [1/3, 2,   1/2, 1,   1,   2,   1  ]
])

def ahp_weights(matrix):
    n=matrix.shape[0]
    eigvals,eigvecs=np.linalg.eig(matrix)
    idx=np.argmax(eigvals.real)
    lambda_max=eigvals.real[idx]
    w=eigvecs[:,idx].real
    w=w/w.sum()
    CI=(lambda_max-n)/(n-1)
    RI_TABLE={1:0.0,2:0.0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
    RI=RI_TABLE.get(n,1.49)
    CR=CI/RI if RI>0 else 0.0
    return w,lambda_max,CI,CR

ahp_w,lambda_max,CI,CR=ahp_weights(AHP_MATRIX)

for name,w in zip(AHP_CRITERIA,ahp_w):
    print(f"  {name:25s}: {w:.4f}")
print("Lambda max:",round(lambda_max,4))
print("Consistency Index (CI):",round(CI,4))
print("Consistency Ratio (CR):",round(CR,4),"-> ACCEPTABLE (<0.10)" if CR<0.1 else "-> INCONSISTENT, revise matrix")

def minmax(series):
    mn,mx=series.min(),series.max()
    if mx>mn: return (series-mn)/(mx-mn)
    return pd.Series(np.zeros(len(series)),index=series.index)

norm_flood=minmax(df_2015["FLOOD_PROBABILITY"])
norm_ugs_need=1-minmax(df_2015["UGS"]) if "UGS" in df_2015.columns else pd.Series(0.5,index=df_2015.index)
norm_imp=minmax(df_2015["IMP"]) if "IMP" in df_2015.columns else minmax(df_2015["ISA"])
norm_wat=minmax(df_2015["WAT"]) if "WAT" in df_2015.columns else minmax(df_2015["WB"])
norm_dem=1-minmax(df_2015["DEM"])
norm_slope=1-minmax(df_2015["Slope"])
norm_urban=minmax(df_2015["urban"]) if "urban" in df_2015.columns else pd.Series(0.5,index=df_2015.index)

criteria_matrix=np.column_stack([norm_flood,norm_ugs_need,norm_imp,norm_wat,norm_dem,norm_slope,norm_urban])
df_2015["GI_SUITABILITY"]=criteria_matrix@ahp_w

plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["GI_SUITABILITY"],
    "AHP Green Infrastructure Suitability","04_AHP_GI_Suitability_Map.png",
    cmap="YlGnBu",vmin=0,vmax=1)

print("\n============================================================")
print("STEP 11 — GI TYPE ALLOCATION")
print("============================================================")

gi_scores=pd.DataFrame({
    "Rain_Garden":0.4*norm_dem+0.3*norm_ugs_need+0.3*norm_slope,
    "Bioswale":0.4*norm_wat+0.3*norm_imp+0.3*norm_slope,
    "Green_Roof":0.5*norm_imp+0.5*norm_urban,
    "Permeable_Pavement":0.4*norm_imp+0.3*norm_urban+0.3*norm_slope
})

df_2015["GI_TYPE"]=gi_scores.idxmax(axis=1)
SUITABILITY_THRESHOLD=0.5
df_2015.loc[df_2015["GI_SUITABILITY"]<SUITABILITY_THRESHOLD,"GI_TYPE"]="Not_Suitable"

gi_type_codes={"Rain_Garden":0,"Bioswale":1,"Green_Roof":2,"Permeable_Pavement":3,"Not_Suitable":4}
df_2015["GI_TYPE_CODE"]=df_2015["GI_TYPE"].map(gi_type_codes)

plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["GI_TYPE_CODE"],
    "GI Type Allocation","05_GI_Type_Allocation_Map.png",
    cmap="tab10",categorical=True,
    cat_labels=["Rain Garden","Bioswale","Green Roof","Permeable Pavement","Not Suitable"])

print("\n============================================================")
print("STEP 12 — NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
print("============================================================")

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize as pymoo_minimize

candidates=df_2015[df_2015["GI_TYPE"]!="Not_Suitable"].reset_index(drop=True)
print("Candidate GI locations:",len(candidates),"of",len(df_2015))

P_arr=candidates["FLOOD_PROBABILITY"].values
ISA_arr=candidates["IMP"].values if "IMP" in candidates.columns else candidates["ISA"].values
BUDGET_FRACTION=0.3

class GIOptimizationProblem(Problem):
    def __init__(self,P,ISA,budget_fraction):
        self.P=P; self.ISA=ISA
        self.n=len(P)
        self.budget=int(budget_fraction*self.n)
        super().__init__(n_var=self.n,n_obj=3,n_constr=1,xl=0,xu=1,type_var=bool)
    def _evaluate(self,X,out,*args,**kwargs):
        X=X.astype(float)
        f1=np.sum(self.P[None,:]*(1-X),axis=1)
        f2=-np.sum(X,axis=1)/self.n
        f3=np.sum(self.ISA[None,:]*(1-X),axis=1)
        g1=np.sum(X,axis=1)-self.budget
        out["F"]=np.column_stack([f1,f2,f3])
        out["G"]=g1.reshape(-1,1)

problem=GIOptimizationProblem(P_arr,ISA_arr,BUDGET_FRACTION)

algorithm=NSGA2(
    pop_size=100,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    eliminate_duplicates=True
)

res=pymoo_minimize(problem,algorithm,("n_gen",100),seed=42,verbose=True)

F=res.F
F_norm=(F-F.min(axis=0))/(F.max(axis=0)-F.min(axis=0)+1e-9)
distances=np.sqrt((F_norm**2).sum(axis=1))
best_idx=int(np.argmin(distances))
best_solution=res.X[best_idx].astype(bool)

candidates["GI_OPTIMIZED"]=best_solution.astype(int)

df_2015=df_2015.merge(candidates[["Row","Column","GI_OPTIMIZED"]],on=["Row","Column"],how="left")
df_2015["GI_OPTIMIZED"]=df_2015["GI_OPTIMIZED"].fillna(0).astype(int)

plot_raster_map(df_2015["Row"],df_2015["Column"],df_2015["GI_OPTIMIZED"],
    "NSGA-II Optimized GI Allocation","06_Optimized_GI_Allocation_Map.png",
    cmap=mcolors.ListedColormap(["#d9d9d9","#2ca25f"]),categorical=True,cat_labels=["No GI","GI Implemented"])

fig=plt.figure(figsize=FIGSIZE)
ax=fig.add_subplot(111,projection="3d")
ax.scatter(res.F[:,0],-res.F[:,1]*100,res.F[:,2],c="tab:red",s=40,label="Pareto solutions")
ax.scatter(res.F[best_idx,0],-res.F[best_idx,1]*100,res.F[best_idx,2],c="tab:blue",s=180,marker="*",label="Selected")
ax.set_xlabel("F1: Residual Flood Risk",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
ax.set_ylabel("F2: Green Coverage (%)",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
ax.set_zlabel("F3: Impervious Exposure",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
ax.set_title("NSGA-II Pareto Front",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
ax.tick_params(labelsize=FONT_SIZE)
for lbl in ax.get_xticklabels()+ax.get_yticklabels()+ax.get_zticklabels():
    lbl.set_fontfamily(FONT_FAMILY); lbl.set_fontsize(FONT_SIZE); lbl.set_fontweight(FONT_WEIGHT)
ax.grid(False)
ax.legend(prop={"family":FONT_FAMILY,"size":FONT_SIZE,"weight":FONT_WEIGHT})
save_map(fig,"07_NSGA_Pareto_Front.png")

print("\n============================================================")
print("STEP 13 — EXISTING vs OPTIMIZED LANDSCAPE")
print("============================================================")

P_all=df_2015["FLOOD_PROBABILITY"].values
ISA_all=df_2015["IMP"].values if "IMP" in df_2015.columns else df_2015["ISA"].values
UGS_all=df_2015["UGS"].values if "UGS" in df_2015.columns else np.zeros(len(df_2015))
GI_all=df_2015["GI_OPTIMIZED"].values
n_total=len(df_2015)

existing_flood_exposure=P_all.sum()
existing_high_risk_pct=100*int((df_2015["RISK_ZONE"]==2).sum())/n_total
existing_green_coverage_pct=100*UGS_all.mean()
existing_imp_exposure=ISA_all.sum()

residual_prob=P_all*(1-GI_all)
optimized_flood_exposure=residual_prob.sum()
optimized_risk_zone=np.array([classify_risk(p) for p in residual_prob])
optimized_high_risk_pct=100*int((optimized_risk_zone==2).sum())/n_total
gi_area_fraction=GI_all.sum()/n_total
optimized_green_coverage_pct=min(100*(UGS_all.mean()+gi_area_fraction),100.0)
optimized_imp_exposure=(ISA_all*(1-GI_all)).sum()

flood_reduction_pct=100*(existing_flood_exposure-optimized_flood_exposure)/existing_flood_exposure if existing_flood_exposure>0 else 0.0

comparison_df=pd.DataFrame({
    "Metric":["High Flood-Risk Area (%)","Green-Space Coverage (%)","Impervious Exposure (sum)","GI Coverage (%)","Flood-Risk Exposure (sum P)"],
    "Existing":[existing_high_risk_pct,existing_green_coverage_pct,existing_imp_exposure,0.0,existing_flood_exposure],
    "Optimized":[optimized_high_risk_pct,optimized_green_coverage_pct,optimized_imp_exposure,100*gi_area_fraction,optimized_flood_exposure]
})
comparison_df["Improvement"]=comparison_df["Optimized"]-comparison_df["Existing"]

comparison_df.to_csv(os.path.join(OUTPUT_FOLDER,"Existing_vs_Optimized_Comparison.csv"),index=False)

fig,ax=plt.subplots(figsize=(13,8))
x_pos=np.arange(len(comparison_df))
width=0.35
ax.bar(x_pos-width/2,comparison_df["Existing"],width,label="Existing",color="#d73027")
ax.bar(x_pos+width/2,comparison_df["Optimized"],width,label="Optimized",color="#2ca25f")
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df["Metric"],rotation=30,ha="right",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
style_axes(ax,"Existing vs Optimized Landscape",xlabel="",ylabel="Value")
ax.legend(prop={"family":FONT_FAMILY,"size":FONT_SIZE,"weight":FONT_WEIGHT})
save_map(fig,"08_Existing_vs_Optimized_Comparison.png")

print("\n============================================================")
print("STEPS 9-13 COMPLETED — Saved in:",OUTPUT_FOLDER,"| dpi:",DPI)
print("============================================================")


# ============================================================
# STEP 15 — EVALUATION MATRICES (each plot own figure/window)
# ALL x-ticks, y-ticks, and legend text = FONT_SIZE (18), BOLD
# ============================================================

from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                              confusion_matrix, accuracy_score)
from sklearn.calibration import calibration_curve

MATRICES_FOLDER="Matrices"
os.makedirs(MATRICES_FOLDER,exist_ok=True)
MDPI=1000

plt.rcParams["font.family"]=FONT_FAMILY
plt.rcParams["font.size"]=FONT_SIZE

def save_fig(fig,name):
    fig.tight_layout()
    fig.savefig(os.path.join(MATRICES_FOLDER,name),dpi=MDPI)
    plt.show()
    plt.close(fig)

def style_plot(ax,title,xlabel,ylabel,legend_loc="best"):
    ax.set_title(title,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.set_xlabel(xlabel,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.tick_params(axis="both",labelsize=FONT_SIZE)
    # BOLD tick numbers as well as bold title/axis labels
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontfamily(FONT_FAMILY)
        lbl.set_fontsize(FONT_SIZE)
        lbl.set_fontweight(FONT_WEIGHT)
    ax.legend(loc=legend_loc,prop={"family":FONT_FAMILY,"size":FONT_SIZE,"weight":FONT_WEIGHT})

tabpfn_train_prob=tabpfn_model.predict_proba(X_train_hybrid)[:,1]
meta_X_train=meta_scaler.transform(np.column_stack([lgbm_train_prob,tabpfn_train_prob]))
hybrid_train_probability=meta_learner.predict_proba(meta_X_train)[:,1]

lgbm_train_pred=(lgbm_train_prob>=lgbm_threshold).astype(int)
tabpfn_train_pred=(tabpfn_train_prob>=tabpfn_threshold).astype(int)
hybrid_train_pred=(hybrid_train_probability>=hybrid_threshold).astype(int)

models_info={
    "LightGBM":{"train_prob":lgbm_train_prob,"val_prob":lgbm_val_prob,"test_prob":lgbm_test_prob,
                "train_pred":lgbm_train_pred,"val_pred":lgbm_val_pred,"test_pred":lgbm_test_pred,"color":"#1f77b4"},
    "TabPFN":{"train_prob":tabpfn_train_prob,"val_prob":tabpfn_val_prob,"test_prob":tabpfn_test_prob,
              "train_pred":tabpfn_train_pred,"val_pred":tabpfn_val_pred,"test_pred":tabpfn_test_pred,"color":"#ff7f0e"},
    "Proposed Hybrid":{"train_prob":hybrid_train_probability,"val_prob":hybrid_val_probability,"test_prob":hybrid_test_probability,
                        "train_pred":hybrid_train_pred,"val_pred":hybrid_val_prediction,"test_pred":hybrid_test_prediction,"color":"#2ca02c"}
}
model_names=list(models_info.keys())

# ---- 1. Model Accuracy: Training vs Validation ----
fig,ax=plt.subplots(figsize=(10,7))
train_acc=[accuracy_score(y_train,models_info[m]["train_pred"]) for m in model_names]
val_acc=[accuracy_score(y_val,models_info[m]["val_pred"]) for m in model_names]
x=np.arange(len(model_names)); width=0.35
ax.bar(x-width/2,train_acc,width,label="Training Accuracy",color="#4c72b0")
ax.bar(x+width/2,val_acc,width,label="Validation Accuracy",color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(model_names,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
ax.set_ylim(0,1)
style_plot(ax,"Model Accuracy — Training vs Validation","Model","Accuracy")
for i,(t,v) in enumerate(zip(train_acc,val_acc)):
    ax.text(i-width/2,t+0.01,f"{t:.3f}",ha="center",fontsize=FONT_SIZE-4,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.text(i+width/2,v+0.01,f"{v:.3f}",ha="center",fontsize=FONT_SIZE-4,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
save_fig(fig,"01_Model_Accuracy_Train_vs_Validation.png")

# ---- 2. Model Loss: Training vs Validation ----
fig,ax=plt.subplots(figsize=(10,7))
rounds_axis2=range(1,len(lgbm_evals_result["train"]["binary_logloss"])+1)
ax.plot(rounds_axis2,lgbm_evals_result["train"]["binary_logloss"],label="Training Loss",color="#4c72b0",linewidth=2)
ax.plot(rounds_axis2,lgbm_evals_result["validation"]["binary_logloss"],label="Validation Loss",color="#dd8452",linewidth=2)
ax.axvline(lgbm_model.best_iteration_,color="black",linestyle="--",label="Best Round")
style_plot(ax,"Model Loss — Training vs Validation (LightGBM)","Boosting Round","Binary Logloss")
save_fig(fig,"02_Model_Loss_Train_vs_Validation.png")

# ---- 3. Calibration Curve ----
fig,ax=plt.subplots(figsize=(10,8))
for m in model_names:
    prob_true,prob_pred=calibration_curve(y_val,models_info[m]["val_prob"],n_bins=10,strategy="uniform")
    ax.plot(prob_pred,prob_true,marker="o",markersize=7,linewidth=2,label=m,color=models_info[m]["color"])
ax.plot([0,1],[0,1],linestyle="--",color="grey",linewidth=2,label="Perfectly Calibrated")
style_plot(ax,"Calibration Curve — Validation Set","Mean Predicted Probability","Fraction of Positives")
save_fig(fig,"03_Calibration_Curve.png")

# ---- 4. ROC Curve (Test set) ----
fig,ax=plt.subplots(figsize=(10,8))
for m in model_names:
    fpr,tpr,_=roc_curve(y_test,models_info[m]["test_prob"])
    roc_auc_val=auc(fpr,tpr)
    lw=3.5 if m=="Proposed Hybrid" else 2
    ax.plot(fpr,tpr,label=f"{m} (AUC={roc_auc_val:.4f})",color=models_info[m]["color"],linewidth=lw)
ax.plot([0,1],[0,1],linestyle="--",color="grey",linewidth=2)
style_plot(ax,"ROC Curve — Test Set","False Positive Rate","True Positive Rate")
save_fig(fig,"04_ROC_Curve_Comparison.png")

# ---- 5. Precision-Recall Curve (Test set) ----
fig,ax=plt.subplots(figsize=(10,8))
for m in model_names:
    precision,recall,_=precision_recall_curve(y_test,models_info[m]["test_prob"])
    pr_auc_val=average_precision_score(y_test,models_info[m]["test_prob"])
    lw=3.5 if m=="Proposed Hybrid" else 2
    ax.plot(recall,precision,label=f"{m} (AP={pr_auc_val:.4f})",color=models_info[m]["color"],linewidth=lw)
style_plot(ax,"Precision-Recall Curve — Test Set","Recall","Precision")
save_fig(fig,"05_PR_Curve_Comparison.png")

# ---- 6. FPR and FNR Bar Plot (Test set) ----
fig,ax=plt.subplots(figsize=(10,7))
fpr_list=[]; fnr_list=[]
for m in model_names:
    tn,fp,fn,tp=confusion_matrix(y_test,models_info[m]["test_pred"],labels=[0,1]).ravel()
    fpr_list.append(fp/(fp+tn) if (fp+tn)>0 else 0.0)
    fnr_list.append(fn/(fn+tp) if (fn+tp)>0 else 0.0)
x=np.arange(len(model_names)); width=0.35
ax.bar(x-width/2,fpr_list,width,label="FPR",color="#c44e52")
ax.bar(x+width/2,fnr_list,width,label="FNR",color="#8172b2")
ax.set_xticks(x)
ax.set_xticklabels(model_names,fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
style_plot(ax,"False Positive Rate vs False Negative Rate — Test Set","Model","Rate")
for i,(f1,f2) in enumerate(zip(fpr_list,fnr_list)):
    ax.text(i-width/2,f1+0.005,f"{f1:.3f}",ha="center",fontsize=FONT_SIZE-4,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
    ax.text(i+width/2,f2+0.005,f"{f2:.3f}",ha="center",fontsize=FONT_SIZE-4,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
save_fig(fig,"06_FPR_FNR_Bar_Plot.png")

# ---- 7. Overall Performance Metrics (Test set) ----
metric_cols=["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC","Specificity","MCC"]
test_results_df=results_df[results_df["Model"].str.contains("Test")].copy()
test_results_df["Model"]=test_results_df["Model"].str.replace(r"\s*\(Test 2010\)","",regex=True)
test_results_df=test_results_df.set_index("Model").loc[["LightGBM","TabPFN","Hybrid LightGBM-TabPFN"]]
test_results_df.index=["LightGBM","TabPFN","Proposed Hybrid"]

fig,ax=plt.subplots(figsize=(15,8))
x=np.arange(len(metric_cols)); n_models=len(test_results_df); width=0.8/n_models
colors_list=[models_info[m]["color"] for m in test_results_df.index]
for i,(mname,row) in enumerate(test_results_df.iterrows()):
    vals=[row[c] for c in metric_cols]
    ax.bar(x+i*width-0.4+width/2,vals,width,label=mname,color=colors_list[i])
ax.set_xticks(x)
ax.set_xticklabels(metric_cols,rotation=30,ha="right",fontsize=FONT_SIZE,fontweight=FONT_WEIGHT,fontfamily=FONT_FAMILY)
style_plot(ax,"Overall Performance Metrics — Test Set","Metric","Score")
save_fig(fig,"07_Overall_Performance_Metrics.png")

test_results_df.to_csv(os.path.join(MATRICES_FOLDER,"Overall_Performance_Metrics.csv"))

print("\n============================================================")
print("STEP 15 — MATRICES COMPLETED")
print("============================================================")
print("Saved in folder:",MATRICES_FOLDER,"| dpi:",MDPI)
for f in sorted(os.listdir(MATRICES_FOLDER)):
    print(" ",f)