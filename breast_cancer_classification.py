"""
================================================================
Breast Cancer Classification using Machine Learning & Deep Learning
================================================================
Author  : Marcos, Carl Ernard M.
Date    : April 2026
Dataset : Wisconsin Breast Cancer Diagnostic (WBCD)
GitHub  : https://github.com/Aerrind/ml-breast-cancer-classification.git
================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)

print("="*65)
print("  Breast Cancer Classification - ML & Deep Learning Pipeline")
print("="*65)

# 1. DATA LOADING
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["diagnosis"] = df["target"].map({0:"Malignant",1:"Benign"})
df.to_csv("/home/claude/breast_cancer_dataset.csv", index=False)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-2} features")
print(f"Benign: {(df.target==1).sum()} | Malignant: {(df.target==0).sum()}")

# 2. EDA FIGURES
mean_cols = [c for c in data.feature_names if "mean" in c]
fig, axes = plt.subplots(1,2,figsize=(13,5))
df["diagnosis"].value_counts().plot.pie(ax=axes[0],autopct="%1.1f%%",startangle=90,
    colors=["#E74C3C","#2ECC71"],shadow=True,textprops={"fontsize":12},explode=[0.04,0])
axes[0].set_title("Class Distribution",fontsize=13,fontweight="bold"); axes[0].set_ylabel("")
corr = df[mean_cols].corr()
mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,ax=axes[1],cmap="RdBu_r",center=0,annot=False,
    linewidths=0.3,square=True,mask=mask)
axes[1].set_title("Feature Correlation (Mean Features)",fontsize=13,fontweight="bold")
axes[1].tick_params(axis="x",rotation=45,labelsize=7)
axes[1].tick_params(axis="y",rotation=0,labelsize=7)
plt.tight_layout(); plt.savefig("/home/claude/fig1_eda.png",dpi=150,bbox_inches="tight"); plt.close()

# 3. PREPROCESSING
X = df.drop(columns=["target","diagnosis"]).values; y = df["target"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 4. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000,C=1.0,random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,learning_rate=0.08,max_depth=4,random_state=42),
    "SVM (RBF)": SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42),
    "Deep Neural Network": MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",
        solver="adam",alpha=1e-4,learning_rate_init=1e-3,max_iter=500,
        early_stopping=True,validation_fraction=0.15,n_iter_no_change=20,random_state=42),
}

# 5. TRAINING & EVALUATION
results = {}
cv_strat = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for name, model in models.items():
    model.fit(X_train_s,y_train)
    y_pred = model.predict(X_test_s); y_prob = model.predict_proba(X_test_s)[:,1]
    cv_acc = cross_val_score(model,X_train_s,y_train,cv=cv_strat,scoring="accuracy").mean()
    results[name] = {"Accuracy":accuracy_score(y_test,y_pred),
        "Precision":precision_score(y_test,y_pred),"Recall":recall_score(y_test,y_pred),
        "F1-Score":f1_score(y_test,y_pred),"ROC-AUC":roc_auc_score(y_test,y_prob),
        "CV-Acc":cv_acc,"y_prob":y_prob,"y_pred":y_pred}
    print(f"  {name:25s} Acc={results[name]['Accuracy']:.4f} AUC={results[name]['ROC-AUC']:.4f} CV={cv_acc:.4f}")

metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV-Acc"]
summary = pd.DataFrame({n:{m:results[n][m] for m in metrics} for n in results}).T.round(4)
summary.to_csv("/home/claude/model_results.csv")
print("\nRESULTS:\n", summary.to_string())

# FIGURES
colors = ["#3498DB","#2ECC71","#E67E22","#9B59B6","#E74C3C"]
fig,ax = plt.subplots(figsize=(8,6))
for (name,res),col in zip(results.items(),colors):
    fpr,tpr,_ = roc_curve(y_test,res["y_prob"])
    ax.plot(fpr,tpr,label=f"{name} (AUC={res['ROC-AUC']:.4f})",color=col,linewidth=2)
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random Classifier")
ax.set_xlabel("False Positive Rate",fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)
ax.set_title("ROC Curves - All Models",fontsize=14,fontweight="bold")
ax.legend(fontsize=9,loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("/home/claude/fig2_roc.png",dpi=150,bbox_inches="tight"); plt.close()

fig,axes = plt.subplots(2,3,figsize=(13,8))
for ax,(name,res) in zip(axes.ravel(),results.items()):
    cm = confusion_matrix(y_test,res["y_pred"])
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,
        xticklabels=["Malignant","Benign"],yticklabels=["Malignant","Benign"],
        cbar=False,linewidths=0.5,annot_kws={"size":14})
    ax.set_title(name,fontsize=10,fontweight="bold")
    ax.set_xlabel("Predicted",fontsize=9); ax.set_ylabel("Actual",fontsize=9)
axes[1,2].set_visible(False)
plt.suptitle("Confusion Matrices - All Models",fontsize=14,fontweight="bold",y=1.01)
plt.tight_layout(); plt.savefig("/home/claude/fig3_cm.png",dpi=150,bbox_inches="tight"); plt.close()

bar_metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
bar_data = summary[bar_metrics]; x = np.arange(len(bar_data)); width = 0.15
bar_palette = ["#2980B9","#27AE60","#E67E22","#8E44AD","#C0392B"]
fig,ax = plt.subplots(figsize=(12,5))
for i,(col,pal) in enumerate(zip(bar_metrics,bar_palette)):
    ax.bar(x+i*width,bar_data[col],width,label=col,color=pal,alpha=0.85)
ax.set_xticks(x+width*2); ax.set_xticklabels(bar_data.index,rotation=12,ha="right",fontsize=9)
ax.set_ylim(0.88,1.005); ax.set_ylabel("Score",fontsize=12)
ax.set_title("Model Performance Comparison",fontsize=14,fontweight="bold")
ax.legend(loc="lower right",fontsize=9,ncol=3); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("/home/claude/fig4_comparison.png",dpi=150,bbox_inches="tight"); plt.close()

rf = models["Random Forest"]
importance_df = pd.DataFrame({"Feature":data.feature_names,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=False).head(10)
fig,ax = plt.subplots(figsize=(9,5))
ax.barh(importance_df["Feature"][::-1],importance_df["Importance"][::-1],color=sns.color_palette("viridis",10))
ax.set_xlabel("Gini Importance",fontsize=12)
ax.set_title("Top-10 Feature Importances (Random Forest)",fontsize=13,fontweight="bold")
ax.grid(axis="x",alpha=0.3)
plt.tight_layout(); plt.savefig("/home/claude/fig5_feature_importance.png",dpi=150,bbox_inches="tight"); plt.close()

print("\nAll figures saved.")
print("\n--- FINAL NUMERIC RESULTS ---")
for name,res in results.items():
    print(f"{name}: Acc={res['Accuracy']:.4f}, P={res['Precision']:.4f}, R={res['Recall']:.4f}, F1={res['F1-Score']:.4f}, AUC={res['ROC-AUC']:.4f}, CV={res['CV-Acc']:.4f}")
