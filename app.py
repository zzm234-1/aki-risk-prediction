"""Streamlit app for the LOCKED 10-predictor AKI CatBoost pipeline.
Place this file beside the supplied main model (model(1).joblib or model.joblib).
Run in the ORIGINAL training environment:
python -m streamlit run enpage_main10_original_visual.py
"""
from pathlib import Path
import hashlib
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from catboost import Pool

st.set_page_config(page_title='AKI Risk Prediction System',layout='wide',page_icon='🏥')
BASE=Path(__file__).resolve().parent
FEATURES=['Cr','BUN','Glu','SpO2','pH','SOFA','Alb','T2DM','SPSIS','RR']

@st.cache_resource
def load_model_and_meta():
    explicit=os.environ.get('AKI_MODEL_PATH')
    paths=[Path(explicit)] if explicit else [BASE/'model(1).joblib',BASE/'model.joblib',BASE/'CatBoost'/'model.joblib']
    for path in paths:
        if not path.is_file(): continue
        pipe=joblib.load(path)
        if not hasattr(pipe,'named_steps') or not {'preprocess','model'}.issubset(pipe.named_steps): continue
        pre=pipe.named_steps['preprocess']; features=list(pre.feature_names_in_)
        if features!=FEATURES: continue
        return pipe,path,hashlib.sha256(path.read_bytes()).hexdigest()
    raise ValueError('Main 10-feature model not found. Put the main model(1).joblib beside this script. The 7-feature sensitivity model is not supported.')

try:
    pipeline,model_path,model_hash=load_model_and_meta()
except Exception as exc:
    st.error(str(exc));st.stop()

st.sidebar.header('Input Clinical Parameters')
with st.sidebar:
    cr=st.number_input('Creatinine, Cr (mg/dL)',min_value=0.01,value=1.0,step=.1,format='%.2f')
    bun=st.number_input('BUN (mg/dL)',min_value=0.0,value=20.0,step=1.0)
    glu=st.number_input('Glucose, Glu (mmol/L)',min_value=0.01,value=5.5,step=.1)
    spo2=st.number_input('Oxygen Saturation, SpO₂ (%)',min_value=1.0,max_value=100.0,value=98.0,step=1.0)
    ph=st.number_input('pH',min_value=6.0,max_value=8.5,value=7.40,step=.01,format='%.2f')
    sofa=st.number_input('Total SOFA Score',min_value=0,max_value=24,value=2,step=1)
    alb=st.number_input('Albumin, Alb (g/dL)',min_value=0.01,value=3.5,step=.1)
    rr=st.number_input('Respiratory Rate, RR (breaths/min)',min_value=1,value=20,step=1)
    t2dm=st.checkbox('Type 2 Diabetes Mellitus (T2DM)',value=False)
    sepsis=st.checkbox('Sepsis (SPSIS)',value=False)


st.title('🏥 AKI Risk Prediction for Elderly Patients with Paralytic Ileus')
st.write("Enter the patient's clinical parameters in the sidebar, then click the button below to obtain the AI-based AKI risk assessment and model explanation.")
submitted=st.button('Start Prediction',type='primary')

values=dict(Cr=cr,BUN=bun,Glu=glu,SpO2=spo2,pH=ph,SOFA=float(sofa),Alb=alb,T2DM=int(t2dm),SPSIS=int(sepsis),RR=float(rr))
input_df=pd.DataFrame([values],columns=FEATURES)


def explain_patient(frame):
    pre=pipeline.named_steps['preprocess']; estimator=pipeline.named_steps['model']
    xt=pre.transform(frame)
    if hasattr(xt,'toarray'): xt=xt.toarray()
    names=[str(n).split('__')[-1] for n in pre.get_feature_names_out()]
    sv=np.asarray(estimator.get_feature_importance(Pool(xt),type='ShapValues'))[0]
    raw=float(np.asarray(estimator.predict(xt,prediction_type='RawFormulaVal')).reshape(-1)[0])
    if len(sv)!=len(names)+1 or not np.isclose(sv.sum(),raw,atol=1e-6):
        raise ValueError('SHAP mapping/additivity check failed.')
    # CatBoost native SHAP, displayed with the ORIGINAL SHAP waterfall renderer.
    # Values are mapped to original-unit inputs in the transformed feature order.
    explanation=shap.Explanation(
        values=sv[:-1],base_values=float(sv[-1]),
        data=frame.iloc[0][names].to_numpy(dtype=float),feature_names=names
    )
    with plt.rc_context({'font.family':'sans-serif','font.size':12}):
        plt.figure(figsize=(10,6))
        shap.plots.waterfall(explanation,max_display=10,show=False)
        fig=plt.gcf()
        fig.set_size_inches(10,6)
        fig.tight_layout()
    return fig

if submitted:
    try:
        probability=float(pipeline.predict_proba(input_df)[0,1])
        st.divider();left,right=st.columns([1,2])
        with left:
            st.subheader('Prediction Result')
            st.metric('Predicted AKI Risk',f'{probability:.2%}')
            st.warning('⚠️ Research model estimate')
            st.write('**Current Input Parameters:**')
            st.dataframe(input_df.T.rename(columns={0:'Value'}),use_container_width=True,height=390)
        with right:
            st.subheader('Model Explanation Using SHAP')
            fig=explain_patient(input_df);st.pyplot(fig);plt.close(fig)
            st.info('💡 Red bars indicate features that increase the predicted AKI risk, whereas blue bars indicate features that decrease the predicted AKI risk.')
            st.caption('SHAP values are expressed in log odds.')
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')


st.markdown('---')
st.caption('Note: This tool is intended for clinical research purposes only and should not be used as the sole basis for clinical diagnosis or treatment decisions.')
