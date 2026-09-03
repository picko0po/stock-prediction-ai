import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='AI Stock Prediction Dashboard', layout='wide')
st.markdown('''<style>.stApp{background:#0b1220}.card{background:#151f31;border:1px solid #2a374d;border-radius:12px;padding:14px}.label{color:#9aa8bd;font-size:13px}.value{font-size:27px;font-weight:700}</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    for n in ['Stock dataset (2).csv','Stock dataset (1).csv','Stock dataset.csv','stock_dataset.csv']:
        try: df=pd.read_csv(n); break
        except FileNotFoundError: continue
    else: st.error('Stock dataset CSV not found beside app.py.'); st.stop()
    df['Date']=pd.to_datetime(df['Date'])
    for c in ['Open','High','Low','Close','Volume','Return','MA_5','MA_20','Volatility_20','Target']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.sort_values(['Ticker','Date']).reset_index(drop=True)

df=load_data(); WINDOW=10
RF_FEATURES=[c for c in ['Open','High','Low','Close','Volume','Return','MA_5','MA_20','Volatility_20'] if c in df]

class StockLSTM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__(); self.lstm=nn.LSTM(1,hidden,1,batch_first=True); self.fc=nn.Linear(hidden,1)
    def forward(self,x):
        o,_=self.lstm(x.unsqueeze(-1)); return self.fc(o[:,-1,:]).squeeze(-1)

def sequences(s):
    s=s.sort_values('Date').dropna(subset=['Return'])
    r=s['Return'].to_numpy(np.float32); d=s['Date'].to_numpy()
    X=[]; y=[]; dates=[]
    for i in range(WINDOW-1,len(r)-2): X.append(r[i-WINDOW+1:i+1]); y.append(r[i+2]); dates.append(d[i])
    return np.asarray(X,np.float32),np.asarray(y,np.float32),pd.to_datetime(dates)

@st.cache_data(show_spinner=False)
def fit_lstm(stock,epochs=30):
    X,y,dates=sequences(stock)
    if len(X)<100:return None
    cut=int(.8*len(X)); Xtr,Xte=X[:cut],X[cut:]; ytr,yte=y[:cut],y[cut:]
    torch.manual_seed(42); m=StockLSTM(); opt=torch.optim.Adam(m.parameters(),lr=.001); loss=nn.HuberLoss(delta=.02)
    tx,ty=torch.tensor(Xtr),torch.tensor(ytr)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); l=loss(m(tx),ty); l.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred=m(torch.tensor(Xte)).numpy()
    out=pd.DataFrame({'Date':dates[cut:],'Actual':yte,'Predicted':pred})
    last=torch.tensor(stock.sort_values('Date')['Return'].dropna().to_numpy(np.float32)[-WINDOW:]).unsqueeze(0)
    with torch.no_grad(): nxt=float(m(last).item())
    return {'r2':r2_score(yte,pred),'mae':mean_absolute_error(yte,pred),'rmse':np.sqrt(mean_squared_error(yte,pred)),'acc':np.mean((yte>=0)==(pred>=0)),'test':out,'next':nxt}

@st.cache_data(show_spinner=False)
def fit_rf(stock):
    x=stock.dropna(subset=RF_FEATURES+['Target']).sort_values('Date'); cut=int(.8*len(x)); tr,te=x.iloc[:cut],x.iloc[cut:]
    m=RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=3,random_state=42,n_jobs=-1); m.fit(tr[RF_FEATURES],tr.Target); p=m.predict(te[RF_FEATURES])
    latest=x.iloc[-1]; nxt=float(m.predict(latest[RF_FEATURES].to_frame().T)[0])
    out=pd.DataFrame({'Date':te.Date,'Actual':te.Target,'Predicted':p})
    fi=pd.DataFrame({'Feature':RF_FEATURES,'Importance':m.feature_importances_}).sort_values('Importance',ascending=False)
    return {'r2':r2_score(te.Target,p),'mae':mean_absolute_error(te.Target,p),'rmse':np.sqrt(mean_squared_error(te.Target,p)),'acc':np.mean((te.Target.to_numpy()>=0)==(p>=0)),'test':out,'next':nxt,'fi':fi}

st.sidebar.title('📈 AI Stock Prediction')
page=st.sidebar.radio('Navigation',['Dashboard','Stock Data','Model Comparison','Predictions','Feature Importance','About'])
tickers=sorted(df.Ticker.unique()); ticker=st.sidebar.selectbox('Select Stock',tickers,index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
s=df[df.Ticker==ticker].copy(); mn,mx=s.Date.min().date(),s.Date.max().date(); dr=st.sidebar.date_input('Date Range',(mn,mx),min_value=mn,max_value=mx)
start,end=dr if isinstance(dr,tuple) and len(dr)==2 else (mn,mx); s=s[(s.Date>=pd.Timestamp(start))&(s.Date<=pd.Timestamp(end))]
if len(s)<120: st.error('Choose a larger date range.'); st.stop()

st.title('AI Stock Prediction Dashboard'); st.caption('LSTM primary model  |  Random Forest comparison')
with st.spinner('Training LSTM and Random Forest...'): lstm=fit_lstm(s); rf=fit_rf(s)
if not lstm: st.error('Not enough data for LSTM.'); st.stop()

def metric_cards():
    vals=[('LSTM R²',lstm['r2']),('LSTM MAE',lstm['mae']),('LSTM RMSE',lstm['rmse']),('LSTM Direction',lstm['acc']*100)]
    cs=st.columns(4)
    for c,(a,v) in zip(cs,vals):
        suffix='%' if 'Direction' in a else ''; c.markdown(f'<div class="card"><div class="label">{a}</div><div class="value">{v:.4f}{suffix}</div></div>',unsafe_allow_html=True)

if page=='Dashboard':
    metric_cards(); st.write('')
    a,b=st.columns([1.5,1])
    with a: st.subheader(f'{ticker} Close Price'); st.line_chart(s.set_index('Date')[['Close']])
    with b: st.subheader('Model R²'); st.bar_chart(pd.DataFrame({'R²':[lstm['r2'],rf['r2']]},index=['LSTM','Random Forest']))
    a,b,c=st.columns(3)
    with a: st.subheader('Direction Accuracy'); st.bar_chart(pd.DataFrame({'Accuracy %':[lstm['acc']*100,rf['acc']*100]},index=['LSTM','Random Forest']))
    with b: st.subheader('RF Feature Importance'); st.bar_chart(rf['fi'].head(9).sort_values('Importance').set_index('Feature'))
    with c:
        st.subheader('Next Prediction'); st.metric('LSTM return',f"{lstm['next']:.4%}"); st.metric('RF return',f"{rf['next']:.4%}"); st.success('LSTM: UP ↑' if lstm['next']>=0 else 'LSTM: DOWN ↓')
    st.subheader('LSTM — Actual vs Predicted Returns'); st.line_chart(lstm['test'].tail(150).set_index('Date')[['Actual','Predicted']])
    st.subheader('Random Forest — Actual vs Predicted Returns'); st.line_chart(rf['test'].tail(150).set_index('Date')[['Actual','Predicted']])
    st.info('LSTM uses the original project design: a 10-return input window and the return two days after the reference date. Results are experimental, not financial advice.')
elif page=='Stock Data':
    st.subheader(f'{ticker} Historical Data'); st.dataframe(s.tail(300),use_container_width=True,hide_index=True)
elif page=='Model Comparison':
    st.subheader('LSTM vs Random Forest'); comp=pd.DataFrame({'Model':['LSTM','Random Forest'],'R²':[lstm['r2'],rf['r2']],'MAE':[lstm['mae'],rf['mae']],'RMSE':[lstm['rmse'],rf['rmse']],'Direction Accuracy':[lstm['acc'],rf['acc']]}); st.dataframe(comp.style.format({'R²':'{:.4f}','MAE':'{:.4f}','RMSE':'{:.4f}','Direction Accuracy':'{:.2%}'}),use_container_width=True,hide_index=True); st.bar_chart(comp.set_index('Model')[['R²','MAE','RMSE']])
elif page=='Predictions':
    st.subheader('Next Prediction'); a,b=st.columns(2); a.metric(' LSTM predicted return',f"{lstm['next']:.4%}"); a.success('UP ↑' if lstm['next']>=0 else 'DOWN ↓'); b.metric('RF predicted return',f"{rf['next']:.4%}"); b.success('UP ↑' if rf['next']>=0 else 'DOWN ↓'); st.subheader('LSTM Predictions vs Actual'); st.line_chart(lstm['test'].tail(150).set_index('Date')[['Actual','Predicted']]); st.dataframe(lstm['test'].tail(40).sort_values('Date',ascending=False),use_container_width=True,hide_index=True)
elif page=='Feature Importance':
    st.subheader('Random Forest Feature Importance'); st.bar_chart(rf['fi'].set_index('Feature')); st.dataframe(rf['fi'],use_container_width=True,hide_index=True)
else:
    st.subheader('About'); st.write('This dashboard uses LSTM as the primary sequence model and Random Forest as the comparison model. The LSTM follows the supplied preprocessors.py design.'); st.write('Available features: '+', '.join(RF_FEATURES))
