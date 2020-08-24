#%%[markdown]
# Sobre Pipeline sklearn objects generator 

#%%
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
# generate some data to play with
X, y = make_classification(
    n_informative=5, n_redundant=0, random_state=42)

#%%
# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5) #THEORY TO BE CHECKED 
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])

# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm = anova_svm.set_params(anova__k=10, svc__C=.1).fit(X[:-2, :], y[:-2])

#%%
import pickle

anova_svm_pkl = pickle.dumps(anova_svm)
#%%
loaded_anova_svm_pkl = pickle.loads(anova_svm_pkl)
#%%
loaded_anova_svm_pkl.predict(X[-2].reshape(1, -1))

# %%
y[-2]

# %%
# generate some data to play with
X, y = make_classification(
    n_informative=5, n_redundant=0, random_state=42)
#%%    
## Ahora versión con 2 pickles, uno preprocesador y otro del modelo
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

scaler = StandardScaler()
scaler_pipeline = Pipeline([('scaler', scaler)])
#%%
'''
Centering and scaling happen independently on each feature by computing the relevant statistics 
on the samples in the training set. Mean and standard deviation are then stored to be used on 
later data using transform.
'''
scaler_pipeline_fitted = scaler_pipeline.fit(X[:-2, :])

#%%
## Ahora obtenemos pickle del scaler
import pickle
import joblib

pickle.dump(scaler_pipeline_fitted, open("preprocessor_pipeline.pickle", "wb"))

#%%
'''
scaler_pipeline_pkl = pickle.dumps(scaler_pipeline_fitted)
loaded_scaler_pipeline_pkl = pickle.loads(scaler_pipeline_pkl)
'''
loaded_preprocessor_pipeline = pickle.load(open("preprocessor_pipeline.pickle", "rb"))

#%%
#scale_training_data
X_train_scaled = loaded_preprocessor_pipeline.transform(X[:-2, :])

#%%
print('X_inference before scaling', X[-2])
X_inference_scaled=loaded_preprocessor_pipeline.transform(X[-2].reshape(1, -1))
print('X_inference after scaling', X_inference_scaled)

#%%[markdown]
# Ahora obtenemos pickle del modelo y entrenamos sobre los datos escalados 
# a partir del pickle pipeline preprocessor
clf = svm.SVC(kernel='linear')
clf_fitted = clf.fit(X_train_scaled, y[:-2])

# %%
#svm_clf_pkl = pickle.dumps(clf_fitted)
pickle.dump(clf_fitted, open("svc_model.pickle", "wb"))
loaded_svm_clf_pkl = pickle.load(open("svc_model.pickle", "rb"))

# %%
loaded_svm_clf_pkl.predict(X_inference_scaled)

# %%
y[-2]

# %%
